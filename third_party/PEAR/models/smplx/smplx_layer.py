# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn
import numpy as np
from .SMPLXV2 import SMPLX
from .lbs import lbs
# from utils.constants import SMPLX_DIR
SMPLX_DIR = 'models'
MEAN_PARAMS = 'models/smpl_mean_params.npz'


class SMPL_Layer(nn.Module):
    """
    Extension of the SMPL Layer with information about the camera for (inverse) projection the camera plane.
    """
    def __init__(self, 
                 type='smplx', 
                 gender='neutral', 
                 num_betas=10,
                 kid=False,
                 person_center=None,
                 *args, 
                 **kwargs,
                 ):
        super().__init__()

        # Args
        assert type == 'smplx'
        self.type = type
        self.kid = kid
        self.num_betas = num_betas

        # self.bm_x = smplx.create(SMPLX_DIR, 'smplx', gender=gender, use_pca=False, flat_hand_mean=True, num_betas=num_betas)
        self.smplx = SMPLX("assets/smplx", n_shape=300, n_exp=50 ,check_pose=True)

        # Primary keypoint - root
        # self.joint_names = eval(f"utils.get_{self.type}_joint_names")()
        # self.person_center = person_center
        # self.person_center_idx = None
        # if self.person_center is not None:
        #     self.person_center_idx = self.joint_names.index(self.person_center)


    def set_smpl_init(self):
        """ Fetch saved SMPL parameters and register buffers."""
        mean_params = np.load(MEAN_PARAMS)
        if self.nrot == 53:
            init_body_pose = torch.eye(3).reshape(1,3,3).repeat(self.nrot,1,1)[:,:,:2].flatten(1).reshape(1, -1)
            init_body_pose[:,:24*6] = torch.from_numpy(mean_params['pose'][:]).float() # global_orient+body_pose from SMPL
        else:
            init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)

        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        init_betas_kid = torch.cat([init_betas, torch.zeros_like(init_betas[:,[0]])],1)
        init_expression = 0. * torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)

        if self.num_betas == 11:
            init_betas = torch.cat([init_betas, torch.zeros_like(init_betas[:,:1])], 1)

        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_betas_kid', init_betas_kid)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_expression', init_expression)


    def forward(self, pose, shape, expression=None, # facial expression
                ):
        """
        Args:
            - pose: pose of the person in axis-angle - torch.Tensor [bs,24,3]
            - shape: torch.Tensor [bs,10]
            - loc: 2D location of the pelvis in pixel space - torch.Tensor [bs,2]
            - dist: distance of the pelvis from the camera in m - torch.Tensor [bs,1]
        Return:
            - dict containing a bunch of useful information about each person
        """

        bs = pose.shape[0]
        out = {}

        # No humans
        if bs == 0:
            return {}
        
        # Low dimensional parameters        
        kwargs_pose = {
            'betas': shape,
        }
        kwargs_pose['global_orient'] = self.bm_x.global_orient.repeat(bs,1)
        kwargs_pose['body_pose'] = pose[:,1:22].flatten(1)
        kwargs_pose['left_hand_pose'] = pose[:,22:37].flatten(1)
        kwargs_pose['right_hand_pose'] = pose[:,37:52].flatten(1)
        kwargs_pose['jaw_pose'] = pose[:,52:53].flatten(1)

        if expression is not None:
            kwargs_pose['expression'] = expression.flatten(1) # [bs,10]
        else:
            kwargs_pose['expression'] = self.bm_x.expression.repeat(bs,1)

        # default - to be generalized
        kwargs_pose['leye_pose'] = self.bm_x.leye_pose.repeat(bs,1)
        kwargs_pose['reye_pose'] = self.bm_x.reye_pose.repeat(bs,1)        
        
        # Forward using the parametric 3d model SMPL-X layer
        # output = self.bm_x(**kwargs_pose)
        # verts = output.vertices # 默认 10475 个角点
        # j3d = output.joints # 前 45 joints 个是标准关键节点   127 点

        new_template_vertices = self.smplx.v_template.unsqueeze(0).repeat(bs, 1, 1)  # [B, 10475, 3]
        full_pose = torch.cat([kwargs_pose['global_orient'],  # [B,1,3]
                               kwargs_pose['body_pose'],  # [B,21,3]
                               kwargs_pose['jaw_pose'],   # [B,1,3]
                               kwargs_pose['leye_pose'] , # [B,2,3]
                               kwargs_pose['reye_pose'],
                               kwargs_pose['left_hand_pose'],  # [B,15,3]
                               kwargs_pose['right_hand_pose'] ], dim=1) # # [B, 1+21+1+2+15+15, 3]

        vertices, joints = lbs(torch.zeros_like(shape_components), full_pose, new_template_vertices,#
                                            self.smplx.shapedirs, self.smplx.posedirs,        # shapedirs[10475, 3, 20]
                                            self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
                                            self.smplx.lbs_weights,joints_offset=joints_offset, dtype=self.smplx.dtype)   # template_vertices（10475x3）


        out.update({
            'v3d': verts, # in 3d camera space
            'j3d': j3d, # in 3d camera space
        })
            
        return out