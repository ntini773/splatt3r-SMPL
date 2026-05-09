import torch
import numpy as np
import torch.nn as nn
from ..mano  import MANO
from ..smplx import SMPLX
from models.smplx.SMPLXV2 import SMPLX as SMPLX_v2
from ..flame import FLAME, vertices2landmarks
from utils import rotation_converter as converter
from ..flame.lbs import lbs,lbs_wobeta,lbs_get_transform, find_dynamic_lmk_idx_and_bcoords,blend_shapes,vertices2joints
from pytorch3d.structures import Meshes
import time
from pytorch3d.transforms import matrix_to_rotation_6d, matrix_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix
class EHM_v2(nn.Module):
    def __init__(self, flame_assets_dir, smplx_assets_dir,
                       n_shape=300, n_exp=50, with_texture=False, add_teeth=True,
                       check_pose=True, use_pca=True, num_pca_comps=6, flat_hand_mean=False, uv_size= 512):
        super().__init__()
        self.smplx = SMPLX(smplx_assets_dir, n_shape=n_shape, n_exp=n_exp, check_pose=check_pose, with_texture = with_texture, add_teeth=add_teeth, uv_size= uv_size)
        self.flame = FLAME(flame_assets_dir, n_shape=n_shape, n_exp=n_exp, with_texture=with_texture, add_teeth=add_teeth) 
        
        v_template,  v_head_template  =  self.smplx.v_template.clone(),  self.flame.v_template.clone()
        tbody_joints = vertices2joints(self.smplx.J_regressor, v_template[None]) # [1,55,3]
        flame_joints = vertices2joints(self.flame.J_regressor, v_head_template[None]) # [5]
        v_template[self.smplx.smplx2flame_ind]=v_head_template - flame_joints[0, 3:5].mean(dim=0, keepdim=True) + tbody_joints[0, 23:25].mean(dim=0, keepdim=True) # [10595,3]
        self.register_buffer('v_template', v_template)
        
        laplacian_matrix = Meshes(verts=[v_template], faces=[self.smplx.faces_tensor]).laplacian_packed().to_dense() 
        self.register_buffer("laplacian_matrix", laplacian_matrix, persistent=False)
        D = torch.diag(laplacian_matrix)
        laplacian_matrix_negate_diag = laplacian_matrix - torch.diag(D) * 2
        self.register_buffer("laplacian_matrix_negate_diag", laplacian_matrix_negate_diag, persistent=False)
        self.get_head_idx_from_pos()
        
    def forward(self, body_param_dict:dict, flame_param_dict:dict=None,mano_param_dict:dict=None, zero_expression=False, zero_jaw=False, zero_shape=False,
                      proj_type='persp', pose_type='rotmat',):
        
        # for flame head model
        if flame_param_dict is not None:
            eye_pose_params    = flame_param_dict['eye_pose_params']     # [2,6]
            shape_params       = flame_param_dict['shape_params']        #.clone()  [B,300]
            expression_params  = flame_param_dict['expression_params']   #.clone()
            global_pose_params = flame_param_dict['pose_params']
            jaw_params         = flame_param_dict['jaw_params']
            eyelid_params      = flame_param_dict['eyelid_params']
            head_scale         = body_param_dict['head_scale']
            

            batch_size = shape_params.shape[0]

            # Adjust shape params size if needed
            if shape_params.shape[1] < self.flame.n_shape:
                shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.flame.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
            
            if zero_expression: 
                expression_params = torch.zeros_like(expression_params).to(shape_params.device)
            if zero_jaw: 
                jaw_params = torch.zeros_like(jaw_params).to(shape_params.device)
            if zero_shape: 
                shape_params = torch.zeros_like(shape_params).to(shape_params.device)

            # eye_pose_params  = self.flame.eye_pose.expand(batch_size, -1)
            neck_pose_params = self.flame.neck_pose.expand(batch_size, -1)

            global_pose_params = torch.zeros_like(global_pose_params).to(shape_params.device)
            neck_pose_params = torch.zeros_like(neck_pose_params).to(shape_params.device)
            # eye_pose_params = torch.zeros_like(eye_pose_params).to(shape_params.device)  # For head test (Teaser  smirk)

            betas = torch.cat([shape_params, expression_params], dim=1)
            full_pose = torch.cat([global_pose_params, neck_pose_params, jaw_params, eye_pose_params], dim=1) # [1,15]

            template_vertices = self.flame.v_template.unsqueeze(0).expand(batch_size, -1, -1) # [1,5023,3]
            
            head_vertices, head_joints, J,T,A = lbs(betas, full_pose, template_vertices, # [1,5023,3]  [1,5,3]
                                             self.flame.shapedirs, self.flame.posedirs,
                                             self.flame.J_regressor, self.flame.parents,
                                             self.flame.lbs_weights, dtype=self.flame.dtype)
            if eyelid_params is not None: # True
                head_vertices = head_vertices + self.flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None] # [:, :self.flame.n_ori_verts]
                head_vertices = head_vertices + self.flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None] # [:, :self.flame.n_ori_verts]
            
            ori_head_vertices = head_vertices.clone()  # [1,5023,3]
            if head_scale is not None:  
                head_vertices = head_vertices * head_scale[:, None]
        else:
            head_vertices = None
            ori_head_vertices = None

        # body paramerters
        shape_params      = body_param_dict['shape']            # torch.Size([1, 200])
        expression_params = body_param_dict['exp']              # torch.Size([1, 50])
        global_pose       = body_param_dict['global_pose']      # torch.Size([1, 1, 3, 3])
        body_pose         = body_param_dict['body_pose']        # torch.Size([1, 21, 3, 3])
        jaw_pose          = body_param_dict.get('jaw_pose',None)         # torch.Size([1, 1, 3, 3])
        left_hand_pose    = body_param_dict['left_hand_pose']   # torch.Size([1, 15, 3, 3])
        right_hand_pose   = body_param_dict['right_hand_pose']  # torch.Size([1, 15, 3, 3])
        eye_pose          = body_param_dict.get('eye_pose',None)         # torch.Size([1, 2, 3, 3])
        joints_offset     = body_param_dict.get('joints_offset',None)
        hand_scale        = body_param_dict['hand_scale']    # torch.Size([1])
        batch_size = shape_params.shape[0]
        
        if expression_params is None: 
            expression_params = self.expression_params.expand(batch_size, -1)
        if global_pose is None: 
            global_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if jaw_pose is None: 
            jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if body_pose is None: 
            body_pose = torch.zeros((batch_size, 21, 3)).to(shape_params.device)
        if len(global_pose.shape) == 2: 
            global_pose = global_pose.unsqueeze(1)
        if len(jaw_pose.shape) == 2: 
            jaw_pose = jaw_pose.unsqueeze(1)

        if len(global_pose.shape) == 3:
            pose2rot = True
            jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
            eye_pose = torch.zeros((batch_size, 2, 3)).to(shape_params.device)
        else:
            pose2rot = False
            jaw_pose = axis_angle_to_matrix(torch.zeros((batch_size, 1, 3)).to(shape_params.device))
            eye_pose = axis_angle_to_matrix(torch.zeros((batch_size, 2, 3)).to(shape_params.device))

        if shape_params.shape[-1] < self.smplx.n_shape:     # 200 < 300  后面 100 维度不重要
            t_shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.smplx.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
        else:
            t_shape_params = shape_params[:, :self.smplx.n_shape] # torch.Size([1, 300])
        
        shape_components = torch.cat([t_shape_params, expression_params], dim=1)  # [1, 300+50]
        
        full_pose = torch.cat([global_pose,  # [B,1,3]
                               body_pose,  # [B,21,3]
                               jaw_pose,   # [B,1,3]
                               eye_pose, # [B,2,3]
                               left_hand_pose,  # [B,15,3]
                               right_hand_pose], dim=1) # # [B, 1+21+1+2+15+15, 3]
        
        template_vertices = self.smplx.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        # 已经做了 shape 变换（beta）
        new_template_vertices = template_vertices + blend_shapes(shape_components, self.smplx.shapedirs) # [1,10475,3]
        tbody_joints = vertices2joints(self.smplx.J_regressor, new_template_vertices)

        if joints_offset is not None: 
            tbody_joints = tbody_joints + joints_offset # [B, 55, 3 ]   

        # new_template_vertices, tbody_joints = lbs(shape_components, torch.zeros_like(full_pose), template_vertices,
        #                                           self.smplx.shapedirs, self.smplx.posedirs,        # shapedirs[10475, 3, 20]
        #                                           self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
        #                                           self.smplx.lbs_weights,joints_offset=joints_offset, dtype=self.smplx.dtype)   # template_vertices（10475x3）
        
        if not hasattr(self, 'head_index'): 
            self.head_index = np.unique(self.flame.head_index)  # [4850]

        if head_vertices is not None: # 把 smplx 的人头顶点替换成 Flame 估计的头
            selected_head = new_template_vertices[:, self.smplx.smplx2flame_ind]  # [B,5023,3]
            ori_selected_head = new_template_vertices[:, self.smplx.smplx2flame_ind].clone()  # [B,5023,3]
            selected_head = head_vertices - head_joints[:, 3:5].mean(dim=1, keepdim=True) + tbody_joints[:, 23:25].mean(dim=1, keepdim=True)
            selected_head[:, self.flame.non_head_index] = ori_selected_head[:, self.flame.non_head_index]       # recover the neck and boundary vertices
            new_template_vertices[:, self.smplx.smplx2flame_ind] = selected_head

        vertices, joints, J, ver_transform_mat, joint_transform_mat = lbs_wobeta( full_pose, new_template_vertices,#
                                            self.smplx.posedirs,       
                                            self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
                                            self.smplx.lbs_weights,joints_offset=joints_offset, pose2rot = pose2rot , dtype=self.smplx.dtype)   # template_vertices（10475x3）
        
        # head_vert = vertices[:, self.smplx.smplx2flame_ind]
        ret_dict = {}

        lmk_faces_idx = self.smplx.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.smplx.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = (
                find_dynamic_lmk_idx_and_bcoords(
                    vertices, full_pose,
                    self.smplx.dynamic_lmk_faces_idx,
                    self.smplx.dynamic_lmk_bary_coords,
                    self.smplx.head_kin_chain)
            )#dyn_lmk_faces_idx([1, 17])
        lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([lmk_bary_coords, dyn_lmk_bary_coords], 1)
        landmarks = vertices2landmarks(vertices, self.smplx.faces_tensor,
                                       lmk_faces_idx,   #  faces_tensor([20908, 3])
                                       lmk_bary_coords)

        final_joint_set = [joints, landmarks]  # [1, 55, 3] [1, 68, 3]
        if hasattr(self.smplx, 'extra_joint_selector'):
            # Add any extra joints that might be needed，extra_joints([1, 22, 3])
            extra_joints = self.smplx.extra_joint_selector(vertices, self.smplx.faces_tensor)
            final_joint_set.append(extra_joints)  # [1, 22, 3]     
        # Create the final joint set  
        joints = torch.cat(final_joint_set, dim=1) # [1, 145, 3]      

        if self.smplx.use_joint_regressor:
            reg_joints = torch.einsum(
                'ji,bik->bjk', self.smplx.extra_joint_regressor, vertices)
            replace_idxs = torch.tensor([2,3,6,7,8,9,10,11,12,13],device=joints.device).long() # [2, 3, 4, ..., 145]
            joints[:, self.smplx.source_idxs[replace_idxs].long()] = ( 
                joints[:, self.smplx.source_idxs[replace_idxs].long()].detach() * 0.0 +
                reg_joints[:, self.smplx.target_idxs[replace_idxs].long()] * 1.0
            )


        # landmarks = torch.cat([landmarks[:, -17:], landmarks[:, :-17]], dim=1)

        # save predcition
        prediction = {  #  [B,10475,3]
            'vertices': vertices,
            'joints': joints,  # [B,145,3]
            'ver_transform_mat':ver_transform_mat, # transform matrix per vertex
            'ori_head_vertices':ori_head_vertices,
        }

        ret_dict.update(prediction)
        
        return ret_dict

    def get_transform_mat(self,body_param_dict:dict, flame_param_dict:dict,mano_param_dict:dict,joints=None):
        # body paramerters
        shape_params      = body_param_dict['shape']                   
        #expression_params = body_param_dict.['exp']
        expression_params=None              
        global_pose       = body_param_dict['global_pose'      ]
        body_pose         = body_param_dict['body_pose'  ]     
        joints_offset     = body_param_dict['joints_offset']
        
        left_hand_pose    = mano_param_dict['left_hand']['hand_pose'   ]
        right_hand_pose   = mano_param_dict['right_hand']['hand_pose' ]
        eye_pose          = flame_param_dict['eye_pose_params']
        jaw_pose         = flame_param_dict['jaw_params']
        
        batch_size = shape_params.shape[0]
        eye_pose=eye_pose.reshape(batch_size,2,3)
        jaw_pose=jaw_pose.reshape(batch_size,1,3).detach().clone()
          
        b, n = left_hand_pose.shape[:2]
        left_hand_pose=converter.batch_matrix2axis(left_hand_pose.flatten(0,1)).reshape(b, n*3)#roma.mappings.rotmat_to_rotvec(batch_mano_left["hand_pose"][:,0,...])[:,None,...]
        right_hand_pose=converter.batch_matrix2axis(right_hand_pose.flatten(0,1)).reshape(b, n, 3)#roma.mappings.rotmat_to_rotvec(batch_mano_right["hand_pose"][:,0,...])[:,None,...] 
        left_hand_pose[:,1::3]*=-1
        left_hand_pose[:,2::3]*=-1
        left_hand_pose=left_hand_pose.reshape(b, n, 3)
        
        if expression_params is None: expression_params = self.smplx.expression_params.expand(batch_size, -1)
        if global_pose is None: global_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if eye_pose is None: eye_pose = torch.zeros((batch_size, 2, 3)).to(shape_params.device)
        if jaw_pose is None: jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if body_pose is None: body_pose = torch.zeros((batch_size, 21, 3)).to(shape_params.device)
        if len(global_pose.shape) == 2: global_pose = global_pose.unsqueeze(1)
        if len(jaw_pose.shape) == 2: jaw_pose = jaw_pose.unsqueeze(1)

        
        if shape_params.shape[-1] < self.smplx.n_shape:
            t_shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.smplx.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
        else:
            t_shape_params = shape_params[:, :self.smplx.n_shape]
        shape_components = torch.cat([t_shape_params, expression_params], dim=1)

        full_pose = torch.cat([global_pose, 
                               body_pose,
                               jaw_pose, 
                               eye_pose,
                               left_hand_pose, 
                               right_hand_pose], dim=1)
        
        template_vertices = self.smplx.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        
        transform_mats, transform_joints = lbs_get_transform(shape_components, full_pose, template_vertices,#
                                            self.smplx.shapedirs, self.smplx.posedirs,        # shapedirs[10475, 3, 20]
                                            self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
                                            self.smplx.lbs_weights,joints_offset=joints_offset,joints=joints, dtype=self.smplx.dtype)   # template_vertices（10475x3）
        return transform_mats,transform_joints
        
    def transform_points3d(self, points3d, M):
        R3d = torch.zeros_like(M)
        R3d[:, :2, :2] = M[:, :2, :2]
        scale = (M[:, 0, 0]**2 + M[:, 0, 1]**2)**0.5
        R3d[:, 2, 2] = scale

        trans = torch.zeros_like(M)[:, 0]
        trans[:, :2] = M[:, :2, 2]
        trans = trans.unsqueeze(1)
        return torch.bmm(points3d, R3d.mT) + trans   # Ugly scale the trans

    def get_head_idx_from_pos(self,y_threshold=0.15):
        y_coordinates = self.v_template[:, 1]
        head_indices_y = torch.where(y_coordinates > y_threshold)[0]
        
        face_vertices=self.v_template[self.smplx.faces_tensor]#f k 3
        face_vertices_nn=face_vertices[self.smplx.uvmap_f_idx.reshape(-1)]# n k 3
        face_bary=self.smplx.uvmap_f_bary.reshape(-1,3)# n k
        
        face_center_nn= torch.einsum('nk,nkj->nj',face_bary,face_vertices_nn)# n 3
        
        head_indices_uv_flat = torch.where(face_center_nn[:,1] > y_threshold)[0]
        
        self.register_buffer('head_idxs_uv_flat',head_indices_uv_flat)
        self.register_buffer('head_idxs_temp',head_indices_y)
