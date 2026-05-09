
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import copy
from omegaconf import OmegaConf
from .pose_transformer import TransformerDecoder
from .smplx_layer import SMPL_Layer
import roma
from .SMPLXV2 import SMPLX
from ..modules.ehm import EHM_v2 
from pytorch3d.transforms import matrix_to_rotation_6d, matrix_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix

def smplx_joints_to_dwpose(joints3d):
    body_mapping = np.array([68, 12, 17, 19, 21, 16, 18, 20, 2, 5,
                             8, 1, 4, 7, 24, 23, 107, 122],
                         dtype=np.int32)
    mapping = [body_mapping]

    lfoot_mapping = np.array([124, 132, 10], dtype=np.int32)
    rfoot_mapping = np.array([135, 143, 11], dtype=np.int32)
    
    mapping += [lfoot_mapping, rfoot_mapping]

    face_contour_mapping = np.arange(106, 106 + 17, dtype=np.int32)
    face_mapping = np.arange(55, 55 + 51, dtype=np.int32)
    mapping += [face_contour_mapping, face_mapping]

    lhand_mapping = np.array([20, 37, 38, 39, 133, 
                                  25, 26, 27, 128, 
                                  28, 29, 30, 129, 
                                  34, 35, 36, 131,
                                  31, 32, 33, 130], dtype=np.int32)
    rhand_mapping = np.array([21, 52, 53, 54, 144, 
                                  40, 41, 42, 139,
                                  43, 44, 45, 140, 
                                  49, 50, 51, 142, 
                                  46, 47, 48, 141], dtype=np.int32)

    mapping += [lhand_mapping, rhand_mapping]

    weights = np.ones([134])
    weights_0  = np.array([0, 1, 14, 15, 16 ,17], dtype=np.int32)
    weights_5  = np.arange(24, 41, dtype=np.int32)
    weights_10 = np.arange(92, 134, dtype=np.int32)
    weights_20 = np.array([3, 4, 6, 7], dtype=np.int32)
    weights_50 = np.array([3, 6], dtype=np.int32)
    weights[weights_0] = 0
    weights[weights_5] = 5
    weights[weights_10] = 10
    weights[weights_20] = 20
    weights[weights_50] = 100

    mapping = np.concatenate(mapping)
    ret_kps3d = joints3d[:, mapping]
    ret_kps3d[:, 1, :2] = (ret_kps3d[:, 2, :2] + ret_kps3d[:, 5, :2]) / 2

    return ret_kps3d, weights

def get_proj_matrix( tanfov,device, z_near=0.01, z_far=100, z_sign=1.0,):

    tanHalfFovY = tanfov
    tanHalfFovX = tanfov

    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right
    z_sign = 1.0

    proj_matrix = torch.zeros(4, 4).float().to("cuda")
    proj_matrix[0, 0] = 2.0 * z_near / (right - left)
    proj_matrix[1, 1] = 2.0 * z_near / (top - bottom)
    proj_matrix[0, 2] = (right + left) / (right - left)
    proj_matrix[1, 2] = (top + bottom) / (top - bottom)
    proj_matrix[3, 2] = z_sign
    proj_matrix[2, 2] = z_sign * z_far / (z_far - z_near)
    proj_matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return proj_matrix


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


class SMPLXTransformerDecoderHead(nn.Module):
    """ 
    Cross-attention based SKEL Transformer decoder
    """

    def __init__(self, cfg, batch_size):
        super().__init__()
        self.cfg = cfg


        n_poses = 318
        self.num_betas = n_betas = 10
        n_cam   = 3
        self.input_is_mean_shape = False
        n_expression  = 10 
        # Build transformer decoder.                                      
        transformer_args = {
                'num_tokens' : 1,
                'token_dim'  : (n_poses + n_betas + n_cam + n_expression) if self.input_is_mean_shape else 1,
                'dim'        : 1024,
            }

        transformer_args.update(OmegaConf.to_container(self.cfg , resolve=True))  # type: ignore
        self.transformer = TransformerDecoder(**transformer_args)

        # Build decoders for parameters.                         
        dim = transformer_args['dim']
        # 'global_pose' 3 , 'body_pose' 21 * 3,  'left_hand_pose' 15 * 3, 'right_hand_pose' 15 * 3 ,  'hand_scale' 3,                              
        self.smplx_poses_decoder = nn.Linear(dim, 312)    # 52 * 3
        self.smplx_scale_decoder = nn.Linear(dim, 6)   
        # 'shape_params' 200   
        self.smplx_shape_decoder = nn.Linear(dim, 200)  
        # 'exp' 50  
        self.smplx_expression_decoder   = nn.Linear(dim, 50)   
        #   'joints_offset' 55 * 3   
        self.smplx_joint_decoder = nn.Linear(dim, 165)


        # 'eye_pose_params' 6. 'pose_params' 3,  'jaw_params' 3, 'eyelid_params' 2, 'head_scale' 3
        self.flame_poses_decoder = nn.Linear(dim, 14)  
        # 'shape_params' 300          
        self.flame_shape_decoder = nn.Linear(dim, 300)     
        #  'expression_params' 50
        self.flame_expression_decoder   = nn.Linear(dim, 50)

        self.cam_decoder   = nn.Linear(dim, n_cam)  # 6 + 3
        # Load mean shape parameters as initial values.
        # from lib.modeling.smplx_model.smpl_layer import SMPL_Layer
        # SMPL Layers
        person_center='head'
        self.nrot = 53
        # self.focal = 24
        self.proj_mats  = None
        self.focal_length = torch.tensor([24])
        self.z_near=0.01
        self.z_far=100

        if self.focal_length.ndim == 1:  # (N,)
            self.focal_length = self.focal_length[:, None]  # (N, 1)
        self.focal_length = self.focal_length.expand(batch_size, 2)  # (N, 2)
        self.set_smpl_init()

    def set_smpl_init(self):
        """ Fetch saved SMPL parameters and register buffers."""
        mean_params = np.load("assets/SMPLX/smpl_mean_params.npz")
        init_body_pose = torch.eye(3).reshape(1,3,3).repeat(self.nrot,1,1)[:,:,:2].flatten(1).reshape(1, -1)
        init_body_pose[:,:24*6] = torch.from_numpy(mean_params['pose'][:]).float() # global_orient+body_pose from SMPL


        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        init_betas_kid = torch.cat([init_betas, torch.zeros_like(init_betas[:,[0]])],1)
        init_expression = 0. * torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)

        if self.num_betas == 11:
            init_betas = torch.cat([init_betas, torch.zeros_like(init_betas[:,:1])], 1)
        # init_R_6d =  [-1,0,0,0,-1,0]
        # init_T = [0,0,0] 

        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_betas_kid', init_betas_kid)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_expression', init_expression)

    def get_projection_transform(self, batch_size=2, device="CUDA"):
        if self.proj_mats is None:
            proj_mats=[]
            if  torch.unique(self.focal_length).numel()==1: # True
                invtanfov=self.focal_length[0,0]  # 
                proj_mat=get_proj_matrix(1/invtanfov,device,z_near=self.z_near,z_far=self.z_far)  # 内参？
                proj_mats=proj_mat[None].repeat(self.focal_length.shape[0],1,1)
            else:
                for invtanfov in self.focal_length:
                    invtanfov=invtanfov[0]; assert invtanfov[0]==invtanfov[1]
                    proj_mat=get_proj_matrix(1/invtanfov,device,z_near=self.z_near,z_far=self.z_far)
                    proj_mats.append(proj_mat[None])
                proj_mats=torch.cat(proj_mats,dim=0)
            self.proj_mats=proj_mats
        else:
            proj_mats=self.proj_mats
        return proj_mats

    def get_full_proj(self, pd_cam):
        B = pd_cam.shape[0]

        T = pd_cam # [B,1,3]
        R = torch.tensor([
                [-1.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0],
                [ 0.0,  0.0,  1.0]
            ], device=pd_cam.device, dtype=pd_cam.dtype).unsqueeze(0).expand(B, -1, -1)


        Tmat=torch.eye(4,device=R.device)[None].repeat(R.shape[0],1,1)
        Tmat[:,:3,:3] = R.clone()
        Tmat[:,:3,3] = T.clone()
        proj_mat=self.get_projection_transform()  # [None].expand(N,-1,-1)  内参
        proj_mat=proj_mat.to(R.device)
        
        B = Tmat.shape[0]

        # 确保 proj_mat 是 [B, 4, 4]
        if proj_mat.dim() == 3:
            proj_mat = proj_mat[0].repeat(B, 1, 1)
        elif proj_mat.shape[0] != B:
            raise ValueError(f"proj_mat batch size {proj_mat.shape[0]} does not match Tmat {B}")

        full_mat = torch.bmm(proj_mat, Tmat)
        return full_mat, Tmat
    
    def rot6d_to_rotmat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to 3x3 rotation matrix.
        Args:
            x: (B, N, 6) 6D representation
        Returns:
            (B, N, 3, 3): rotation matrices
        """
        B, N = x.shape[:2]
        x = x.view(B, N, 2, 3)  # (B, N, 2, 3)

        a1 = x[:, :, 0]  # (B, N, 3)
        a2 = x[:, :, 1]  # (B, N, 3)

        b1 = F.normalize(a1, dim=-1)  # (B, N, 3)
        dot = torch.einsum('bij,bij->bi', b1, a2).unsqueeze(-1)  # (B, N, 1)
        b2 = F.normalize(a2 - dot * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        rotmat = torch.stack([b1, b2, b3], dim=-1)  # (B, N, 3, 3)
        return rotmat

    def forward(self, x, **kwargs):
        B = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        # Input token to transformer is zero token.
        # with PM.time_monitor('init_token'):
        token = x.new_zeros(B, 1, 1) # here

        # Pass through transformer.
        # with PM.time_monitor('transformer'):
        token_out = self.transformer(token, context=x)  #  (refined tokens)
        token_out = token_out.squeeze(1)  # (B, C)


        # 'eye_pose_params' 6. 'pose_params' 3,  'jaw_params' 3, 'eyelid_params' 2, 'head_scale' 3  
        flame_param_dict = {}
        flame_pose = self.flame_poses_decoder(token_out)
        flame_param_dict['eye_pose_params'] = flame_pose[:,:6]
        flame_param_dict['pose_params'] = flame_pose[:,6:9]
        flame_param_dict['jaw_params'] = flame_pose[:,9:12]
        flame_param_dict['eyelid_params'] = flame_pose[:,12:14]
        flame_param_dict['expression_params']  = self.flame_expression_decoder(token_out) 
        flame_param_dict['shape_params'] = self.flame_shape_decoder(token_out)


        # 'global_pose' 3 , 'body_pose' 21 * 3,  'left_hand_pose' 15 * 3, 'right_hand_pose' 15 * 3 ,  'hand_scale' 3,                              
        #  self.smplx_poses_decoder = nn.Linear(dim, 159)     55 * 3  55 * 6
        body_param_dict = {}
        smplx_pose = self.smplx_poses_decoder(token_out) 
        smplx_pose[:,:132] +=  self.init_body_pose[:,:132]
        body_param_dict['global_pose'] = self.rot6d_to_rotmat( smplx_pose[:,:6].unsqueeze(1).reshape((-1,1,6))  )   # * [3, 1, 1]
        body_param_dict['body_pose'] =  self.rot6d_to_rotmat( smplx_pose[:,6:132].unsqueeze(1).reshape((-1,21,6)) )  # * 0.5
        body_param_dict['left_hand_pose'] = self.rot6d_to_rotmat( smplx_pose[:,132:222].unsqueeze(1).reshape((-1,15,6)) )
        body_param_dict['right_hand_pose'] = self.rot6d_to_rotmat( smplx_pose[:,222:312].unsqueeze(1).reshape((-1,15,6)) )


        smplx_scale = self.smplx_scale_decoder(token_out)
        body_param_dict['hand_scale'] = smplx_scale[:,:3] 
        body_param_dict['head_scale'] = smplx_scale[:,3:]

        body_param_dict['eye_pose'] = None
        body_param_dict['jaw_pose'] = None
        body_param_dict['joints_offset'] = None # self.smplx_joint_decoder(token_out).reshape(-1,55,3)
        body_param_dict['exp'] = self.smplx_expression_decoder(token_out) 
        body_param_dict['shape'] = self.smplx_shape_decoder(token_out)


        bias = torch.tensor([ 0, 0, 1.5], device=token_out.device)
        pd_cam = self.cam_decoder(token_out)
        pd_cam += bias
        pd_cam[:, 2:] =  24 /  (pd_cam[:, 2:] +  1e-9) # f / s


        full_project, RT= self.get_full_proj(pd_cam) # [B,9]   

        # rotvec = roma.rotmat_to_rotvec(pd_poses)  


        all_out =  {}
        all_out['pd_cam'] = RT  # [4,4]
        all_out['body_param'] = body_param_dict
        all_out['flame_param'] = flame_param_dict

        return all_out






