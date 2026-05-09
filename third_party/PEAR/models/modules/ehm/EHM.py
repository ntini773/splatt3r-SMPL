import torch
import numpy as np
import torch.nn as nn
from ..mano  import MANO
from ..smplx import SMPLX
from ..flame import FLAME, vertices2landmarks
from utils import rotation_converter as converter
from ..flame.lbs import lbs,lbs_wobeta,lbs_get_transform, find_dynamic_lmk_idx_and_bcoords,blend_shapes,vertices2joints
import time

class EHM(nn.Module):
    def __init__(self, flame_assets_dir, smplx_assets_dir, mano_assets_dir,
                       n_shape=300, n_exp=50, with_texture=False, 
                       check_pose=True, use_pca=True, num_pca_comps=6, flat_hand_mean=False,add_teeth=False):
        super().__init__()
        self.smplx = SMPLX(smplx_assets_dir, n_shape=n_shape, n_exp=n_exp, check_pose=check_pose, with_texture=with_texture,add_teeth=add_teeth)
        self.flame = FLAME(flame_assets_dir, n_shape=n_shape, n_exp=n_exp, with_texture=with_texture,add_teeth=add_teeth)
        self.mano  = MANO(mano_assets_dir, use_pca=use_pca, num_pca_comps=num_pca_comps, flat_hand_mean=flat_hand_mean)

    
    def forward(self, body_param_dict:dict, flame_param_dict:dict=None, mano_param_dict:dict=None, zero_expression=False, zero_jaw=False, zero_shape=False,
                       pose_type='rotmat',):
        
        # for flame head model
        start_time=time.time()
        if flame_param_dict is not None:
            eye_pose_params    = flame_param_dict['eye_pose_params']# batch_size,6
            shape_params       = flame_param_dict['shape_params']# batch_size,300
            expression_params  = flame_param_dict['expression_params']# batch_size,50
            global_pose_params = flame_param_dict.get('pose_params', None)# batch_size,3
            jaw_params         = flame_param_dict.get('jaw_params', None)# batch_size,3
            eyelid_params      = flame_param_dict.get('eyelid_params', None) ## batch_size,2
            head_scale         = body_param_dict.get('head_scale', None) # batch_size
            head_pos_offset    = body_param_dict.get('head_pos_offset', None) # batch_size 3

            batch_size = shape_params.shape[0]

            # Adjust shape params size if needed
            if shape_params.shape[1] < self.flame.n_shape:
                shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.flame.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
            
            if zero_expression: expression_params = torch.zeros_like(expression_params,device=shape_params.device)
            if zero_jaw: jaw_params = torch.zeros_like(jaw_params,device=shape_params.device)
            if zero_shape: shape_params = torch.zeros_like(shape_params,device=shape_params.device)

            # eye_pose_params  = self.flame.eye_pose.expand(batch_size, -1)
            neck_pose_params = self.flame.neck_pose.expand(batch_size, -1)

            global_pose_params = torch.zeros_like(global_pose_params,device=shape_params.device)
            neck_pose_params = torch.zeros_like(neck_pose_params,device=shape_params.device)

            betas = torch.cat([shape_params, expression_params], dim=1)
            full_pose = torch.cat([global_pose_params, neck_pose_params, jaw_params, eye_pose_params], dim=1)

            template_vertices = self.flame.v_template.unsqueeze(0).expand(batch_size, -1, -1)
            print(f"Flame_mid_time:{(time.time()-start_time)*1000}ms")
            head_vertices, head_joints = lbs(betas, full_pose, template_vertices,
                                             self.flame.shapedirs, self.flame.posedirs,
                                             self.flame.J_regressor, self.flame.parents,
                                             self.flame.lbs_weights, dtype=self.flame.dtype)
            
            if eyelid_params is not None:
                head_vertices = head_vertices + self.flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None] #[:, :self.flame.n_ori_verts]
                head_vertices = head_vertices + self.flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]#[:, :self.flame.n_ori_verts]
            head_vertices=head_vertices*head_scale[:,None,None]+head_pos_offset[:,None]
            
        else:
            head_vertices = None
        print(f"Flame_time:{(time.time()-start_time)*1000}ms")
        start_time=time.time()
        if mano_param_dict is not None:
            
            left_hand_scale    = body_param_dict.get('left_hand_scale', None) # batch_size
            left_hand_pos_offset    = body_param_dict.get('left_hand_pos_offset', None) # batch_size,3
            right_hand_scale   = body_param_dict.get('right_hand_scale', None) # batch_size
            right_hand_pos_offset   = body_param_dict.get('right_hand_pos_offset', None)# batch_size,3
            # hand parameters
            mano_param_dict_left = mano_param_dict['left_hand']
            betas = mano_param_dict_left.get('betas', self.mano.betas)# batch_size 10 3
            hand_pose     = mano_param_dict_left.get('hand_pose', self.mano.hand_pose)# batch_size 15 3

            b, n = hand_pose.shape[:2]
            global_orient = torch.zeros(b, 1, 3).to(hand_pose.device)
            if pose_type == 'rotmat': hand_pose = converter.batch_matrix2axis(hand_pose.flatten(0, 1)).reshape(b, n, 3)

            if global_orient.shape[-2] == 3 and global_orient.shape[-1] == 3:
                global_orient = converter.batch_matrix2axis(global_orient).unsqueeze(1)
            if hand_pose.shape[-2] == 3 and hand_pose.shape[-1] == 3:
                b, n = hand_pose.shape[:2]
                hand_pose = converter.batch_matrix2axis(hand_pose.flatten(0,1)).reshape(b, n, 3)

            full_pose = torch.cat([global_orient, hand_pose], dim=1)
            if betas.shape[0] != full_pose.shape[0]: 
                t_betas = betas[0].unsqueeze(0).expand(full_pose.shape[0], -1)
            else:
                t_betas = betas
            left_hand_vertices, left_hand_joints = lbs(t_betas, full_pose, self.mano.v_template,
                                                    self.mano.shapedirs, self.mano.posedirs,
                                                    self.mano.J_regressor, self.mano.parents,
                                                    self.mano.lbs_weights)
            left_hand_vertices=left_hand_vertices*left_hand_scale[:,None,None]+left_hand_pos_offset[:,None]
            mano_param_dict_right = mano_param_dict['right_hand']
            betas = mano_param_dict_right.get('betas', self.mano.betas)# batch_size 10 3
            global_orient = mano_param_dict_right.get('global_orient', self.mano.global_orient)
            hand_pose     = mano_param_dict_right.get('hand_pose', self.mano.hand_pose)# batch_size 15 3

            b, n = hand_pose.shape[:2]
            global_orient = torch.zeros(b, 1, 3).to(hand_pose.device)
            if pose_type == 'rotmat': hand_pose = converter.batch_matrix2axis(hand_pose.flatten(0, 1)).reshape(b, n, 3)

            if global_orient.shape[-2] == 3 and global_orient.shape[-1] == 3:
                global_orient = converter.batch_matrix2axis(global_orient).unsqueeze(1)
            if hand_pose.shape[-2] == 3 and hand_pose.shape[-1] == 3:
                b, n = hand_pose.shape[:2]
                hand_pose = converter.batch_matrix2axis(hand_pose.flatten(0,1)).reshape(b, n, 3)

            full_pose = torch.cat([global_orient, hand_pose], dim=1)

            if betas.shape[0] != full_pose.shape[0]: 
                t_betas = betas[0].unsqueeze(0).expand(full_pose.shape[0], -1)
            else:
                t_betas = betas
            right_hand_vertices, right_hand_joints = lbs(t_betas, full_pose, self.mano.v_template,
                                                        self.mano.shapedirs, self.mano.posedirs,
                                                        self.mano.J_regressor, self.mano.parents,
                                                        self.mano.lbs_weights)
            right_hand_vertices=right_hand_vertices*right_hand_scale[:,None,None]+right_hand_pos_offset[:,None]
        else:
            left_hand_vertices = right_hand_vertices = None
        print(f"Mano_time:{(time.time()-start_time)*1000}ms")
        start_time=time.time()
        # body paramerters
        shape_params      = body_param_dict.get('shape')                   # torch.Size([1, 250])
        expression_params = body_param_dict.get('exp', None)               # torch.Size([1, 50])
        global_pose       = body_param_dict.get('global_pose', None)       # torch.Size([1, 1, 3, 3])
        body_pose         = body_param_dict.get('body_pose', None)         # torch.Size([1, 21, 3, 3])
        jaw_pose          = body_param_dict.get('jaw_pose', None)          # torch.Size([1, 1, 3, 3])
        left_hand_pose    = body_param_dict.get('left_hand_pose', None)    # torch.Size([1, 15, 3, 3])
        right_hand_pose   = body_param_dict.get('right_hand_pose', None)   # torch.Size([1, 15, 3, 3])
        eye_pose          = body_param_dict.get('eye_pose', None)          # torch.Size([1, 2, 3, 3])
        joints_offset     = body_param_dict.get('joints_offset',None)     # batch_size 55 3
        batch_size = shape_params.shape[0]
        


        if expression_params is None: expression_params = self.expression_params.expand(batch_size, -1)
        if global_pose is None: global_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        # if jaw_pose is None: jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if body_pose is None: body_pose = torch.zeros((batch_size, 21, 3)).to(shape_params.device)
        if len(global_pose.shape) == 2: global_pose = global_pose.unsqueeze(1)
        # if len(jaw_pose.shape) == 2: jaw_pose = jaw_pose.unsqueeze(1)
        
            
        jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        eye_pose = torch.zeros((batch_size, 2, 3)).to(shape_params.device)
        left_hand_pose  = torch.zeros((batch_size, 15, 3)).to(shape_params.device)
        right_hand_pose = torch.zeros((batch_size, 15, 3)).to(shape_params.device)

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
        
        new_template_vertices = template_vertices + blend_shapes(shape_components, self.smplx.shapedirs)
        tbody_joints = vertices2joints(self.smplx.J_regressor, new_template_vertices)
        if joints_offset is not None: tbody_joints=tbody_joints+joints_offset


        if not hasattr(self, 'head_index'): self.head_index = np.unique(self.flame.head_index)
        if head_vertices is not None:
            selected_head = new_template_vertices[:, self.smplx.smplx2flame_ind]
            selected_head[:, self.head_index] = head_vertices[:, self.head_index] - head_joints[:, 3:5].mean(dim=1, keepdim=True) + tbody_joints[:, 23:25].mean(dim=1, keepdim=True)
            new_template_vertices[:, self.smplx.smplx2flame_ind] = selected_head
        if left_hand_vertices is not None:
            t_ver = left_hand_vertices - left_hand_joints[:, 0:1, :]
            t_ver[..., 0] = -t_ver[..., 0]
            new_template_vertices[:, self.smplx.smplx2mano_ind['left_hand']] = t_ver + tbody_joints[:, 20:21, :]
        if right_hand_vertices is not None:
            new_template_vertices[:, self.smplx.smplx2mano_ind['right_hand']] = right_hand_vertices - right_hand_joints[:, 0:1, :] + tbody_joints[:, 21:22, :]

        # vertices, joints = lbs(torch.zeros_like(shape_components), full_pose, new_template_vertices,#
        #                                     self.smplx.shapedirs, self.smplx.posedirs,        # shapedirs[10475, 3, 20]
        #                                     self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
        #                                     self.smplx.lbs_weights,joints_offset=joints_offset, dtype=self.smplx.dtype)   # template_vertices（10475x3）
        vertices, joints_transform,joints,ver_transform_mat,joint_transform_mat = lbs_wobeta( full_pose, new_template_vertices,#
                                            self.smplx.posedirs,       
                                            self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
                                            self.smplx.lbs_weights,joints_offset=joints_offset, dtype=self.smplx.dtype)   # template_vertices（10475x3）
        #head_vert = vertices[:, self.smplx.smplx2flame_ind]
        ret_dict = {}
        print(f"smplx_time:{(time.time()-start_time)*1000}ms")
        
        # face dynamic landmarks，lmk_faces_idx（51），lmk_bary_coords（51x3）
        # lmk_faces_idx = self.smplx.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        # lmk_bary_coords = self.smplx.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        # dyn_lmk_faces_idx, dyn_lmk_bary_coords = (
        #         find_dynamic_lmk_idx_and_bcoords(
        #             vertices, full_pose,
        #             self.smplx.dynamic_lmk_faces_idx,
        #             self.smplx.dynamic_lmk_bary_coords,
        #             self.smplx.head_kin_chain)
        #     )#dyn_lmk_faces_idx([1, 17])
        # lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
        # lmk_bary_coords = torch.cat([lmk_bary_coords, dyn_lmk_bary_coords], 1)
        # landmarks = vertices2landmarks(vertices, self.smplx.faces_tensor,
        #                                lmk_faces_idx,#faces_tensor([20908, 3])
        #                                lmk_bary_coords)
        
        # final_joint_set = [joints, landmarks]
        # if hasattr(self.smplx, 'extra_joint_selector'):
        #     # Add any extra joints that might be needed，extra_joints([1, 22, 3])
        #     extra_joints = self.smplx.extra_joint_selector(vertices, self.smplx.faces_tensor)
        #     final_joint_set.append(extra_joints)
        # # Create the final joint set
        # joints = torch.cat(final_joint_set, dim=1)
        # if self.smplx.use_joint_regressor:
        #     reg_joints = torch.einsum(
        #         'ji,bik->bjk', self.smplx.extra_joint_regressor, vertices)

        #     joints[:, self.smplx.source_idxs.long()] = (
        #         joints[:, self.smplx.source_idxs.long()].detach() * 0.0 +
        #         reg_joints[:, self.smplx.target_idxs.long()] * 1.0
        #     )

        # landmarks = torch.cat([landmarks[:, -17:], landmarks[:, :-17]], dim=1)


        # save predcition
        prediction = {
            'vertices': vertices,
            # 'face_kpt': landmarks,
            'joints': joints,                  # tpose joints
            'joints_transform':joints_transform,#transformed joints
            'ver_transform_mat':ver_transform_mat, # transform matrix per vertex
            'joint_transform_mat':joint_transform_mat, # transofrm matrix per joint
            'head_vertices': vertices[:, self.smplx.smplx2flame_ind][:, self.head_index],
            'head_ref_joint': joints[:, 23:25].mean(dim=1, keepdim=True),

            'left_hand_vertices': vertices[:, self.smplx.smplx2mano_ind['left_hand']],
            'left_hand_ref_joint': joints[:, 20:21, :],

            'right_hand_vertices': vertices[:, self.smplx.smplx2mano_ind['right_hand']],
            'right_hand_ref_joint': joints[:, 21:22, :],
        }

        ret_dict.update(prediction)
        
        return ret_dict

    def get_transform_mat(self,body_param_dict:dict, flame_param_dict:dict,mano_param_dict:dict,joints=None):
        # body paramerters
        shape_params      = body_param_dict.get('shape')                   
        #expression_params = body_param_dict.get('exp', None)
        expression_params=None              
        global_pose       = body_param_dict.get('global_pose', None)       
        body_pose         = body_param_dict.get('body_pose', None)        
        joints_offset     = body_param_dict.get('joints_offset',None)
        
        left_hand_pose    = mano_param_dict['left_hand'].get('hand_pose', None)    
        right_hand_pose   = mano_param_dict['right_hand'].get('hand_pose', None)  
        eye_pose          = flame_param_dict.get('eye_pose_params', None)
        jaw_pose         = flame_param_dict.get('jaw_params', None)
        
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


