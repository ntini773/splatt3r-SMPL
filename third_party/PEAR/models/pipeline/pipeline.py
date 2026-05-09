import pytorch_lightning as pl
from models.backbones import ViT
from models.smplx.smplx_head import SMPLXTransformerDecoderHead
from models.pipeline.loss import Keypoint2DLoss, Keypoint3DLoss,  HeadParameterLoss, BodyParameterLoss, CameraLoss, ParameterLoss
import lightning
from lightning.fabric.strategies import DDPStrategy
import torch
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, List
import torch
import numpy as np 
from utils.pipeline_utils import *
import os,shutil
import lightning
from lightning.fabric.strategies import DDPStrategy
import torch,torchvision
import numpy as np
from datetime import datetime
from tqdm import tqdm
from utils.general_utils import (
    ConfigDict, rtqdm, device_parser, 
    calc_parameters, biuld_logger
)

import pickle
import random
import cv2
from models.smplx.smplx_utils import smplx_joints_to_dwpose, smplx_to_dwpose
from ..modules.ehm import EHM_v2 
from utils.graphics_utils import GS_Camera
import imageio
from utils.draw import draw_landmarks
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from pytorch3d.renderer import PointLights
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from utils.smplx2smpl_joints import smplx2smpl_joints
from smplx import SMPL, SMPLX
import joblib
import pickle
from utils.vis_mesh import smpl_para2mesh

class OurPipeline():
    def __init__(self, cfg, train_dataloader,val_dataloader,devices, name:str="Test", init_backbone=True):
        super(OurPipeline, self).__init__()
        self.name = name
        self.cfg = cfg
        self._dump_dir =  os.path.join('outputs',  datetime.now().strftime("%Y%m%d_%H"))
        self._writer_dir =  os.path.join('outputs', datetime.now().strftime("%Y%m%d_%H"), 'writers')
        self._total_iters = cfg.TRAIN.train_iter
        self._check_interval = cfg.TRAIN.check_interval 
        self._visual_train_interval = 1000 
        self._debug = False
        self.body_image_size = 1024
        self.head_image_size = 512
        self.device = devices
        self.backbone = ViT(**cfg.BACKBONE)
        self.head = SMPLXTransformerDecoderHead(cfg.HEAD, cfg.TRAIN.batch_size)
        self.optimizer = self.configure_optimizers()  # backbone  and  decoder head  


        self.loss_weight = {'kp3d': 0.05, 'kp2d': 0.01, 'poses_orient': 0.002, 'poses_body': 0.001, 'betas': 0.0005}


        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if init_backbone:
            # For inference mode with tuned backbone checkpoints, we don't need to initialize the backbone here.
            self._init_backbone()

        # build trainer
        self.lightning_fabric = lightning.Fabric(
            accelerator='cuda', strategy= DDPStrategy(find_unused_parameters=True), devices=devices, # 6.26 find_unused_parameters = true ， precision='16-mixed',  "auto" 
        )
        self.lightning_fabric.launch()

        self.backbone, self.optimizer = self.lightning_fabric.setup(self.backbone, self.optimizer)
        self.head = self.lightning_fabric.setup(self.head)


        self.train_dataloader = self.lightning_fabric.setup_dataloaders(train_dataloader,use_distributed_sampler=True)
        self.val_dataloader = self.lightning_fabric.setup_dataloaders(val_dataloader,use_distributed_sampler=True)  #use_distributed_sampler=self.use_distributed_sampler


        cameras_kwargs = self.build_cameras_kwargs(cfg.TRAIN.batch_size,focal_length=24)
        self.cameras = GS_Camera(**cameras_kwargs).to(self.lightning_fabric.device)


        self.ehm = EHM_v2( "assets/FLAME", "assets/SMPLX").to(self.lightning_fabric.device)  # 12.18 revised
        self.smplx=self.ehm.smplx
        self.v_template=torch.nn.Parameter(self.ehm.v_template,requires_grad=False)
        self.body_renderer = BodyRenderer("assets/SMPLX", self.body_image_size , focal_length=24.0 ).to(self.lightning_fabric.device) 



        self.smplx2smpl = torch.from_numpy(joblib.load("assets/SMPLX2SMPL/body_models/smplx2smpl.pkl")['matrix']).unsqueeze(0).float().to(self.lightning_fabric.device) 
        self.smpl = SMPL("assets/SMPL/SMPL_NEUTRAL.pkl", gender='neutral').to(self.lightning_fabric.device) 

        self.J_regressor_extra = torch.tensor(pickle.load( open("assets/SMPLX2SMPL/SMPL_to_J19.pkl", 'rb'), # SMPL_to_J19.pkl
                            encoding='latin1'), dtype=torch.float32).to(self.lightning_fabric.device) 
        

        # Loss layers.
        self.body_params_loss = BodyParameterLoss() 
        self.head_params_loss = HeadParameterLoss()
        self.camera_loss = CameraLoss()

        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')

        self.params_loss = ParameterLoss()

        # .to(self.lightning_fabric.device)
        # self.lmk2d_loss = Landmark2DLoss(self.ehm.smplx.lmk_203_left_indices,
        #                                  self.ehm.smplx.lmk_203_right_indices,
        #                                  self.ehm.smplx.lmk_203_front_indices,
        #                                  self.ehm.smplx.lmk_mp_indices, metric='l1').to(self.lightning_fabric.device)
        self.metric = torch.nn.L1Loss(reduction = 'sum') # .to(self.lightning_fabric.device)



        # Manually control the optimization since we have an adversarial process.  
        self.automatic_optimization = False
        self.set_data_adaption()

        # For visualization debug.
        if False:
            self.wis3d = Wis3D(seq_name=PM.cfg.exp_name)
        else:
            self.wis3d = None


    def forward(self, batch):
        '''
        ### Returns
        - outputs: Dict
            - pd_kp3d: torch.Tensor, shape (B, Q=44, 3)
            - pd_kp2d: torch.Tensor, shape (B, Q=44, 2)
            - pred_keypoints_2d: torch.Tensor, shape (B, Q=44, 2)
            - pred_keypoints_3d: torch.Tensor, shape (B, Q=44, 3)
            - pd_params: Dict
                - poses: torch.Tensor, shape (B, 46)
                - betas: torch.Tensor, shape (B, 10)
            - pd_cam: torch.Tensor, shape (B, 3)
            - pd_cam_t: torch.Tensor, shape (B, 3)
            - focal_length: torch.Tensor, shape (B, 2)
        '''
        batch = self.adapt_batch(batch)

        # 1. Main parts forward pass.
        img_patch = to_tensor(batch['img_patch'], self.device)  # (B, C, H, W)
        outputs = self.forward_step(img_patch)  # {...}

        # 2. Prepare the secondary products
        # 2.1. Body model outputs.
        pd_skel_params = OurPipeline._adapt_skel_params(outputs['pd_params'])
        skel_outputs = self.skel_model(**pd_skel_params, skelmesh=False)
        pd_kp3d = skel_outputs.joints  # (B, Q=44, 3)
        pd_skin_verts = skel_outputs.skin_verts.detach().cpu().clone()  # (B, V=6890, 3)
        # 2.2. Reproject the 3D joints to 2D plain.
        pd_kp2d = perspective_projection(
                points       = to_tensor(pd_kp3d, device=self.device),  # (B, K=Q=44, 3)
                translation  = to_tensor(outputs['pd_cam_t'], device=self.device),  # (B, 3)
                focal_length = to_tensor(outputs['focal_length'], device=self.device) / self.cfg.policy.img_patch_size,  # (B, 2)
            )

        outputs['pd_kp3d'] = pd_kp3d
        outputs['pd_kp2d'] = pd_kp2d
        outputs['pred_keypoints_2d'] = pd_kp2d  # adapt HMR2.0's script
        outputs['pred_keypoints_3d'] = pd_kp3d  # adapt HMR2.0's script
        outputs['pd_params'] = pd_skel_params
        outputs['pd_skin_verts'] = pd_skin_verts

        return outputs

    def forward_step(self, x:torch.Tensor):
        '''
        Run an inference step on the model.

        ### Args
        - x: torch.Tensor, shape (B, C, H, W)
            - The input image patch.

        ### Returns
        - outputs: Dict
            - 'pd_cam': torch.Tensor, shape (B, 3)
                - The predicted camera parameters.
            - 'pd_params': Dict
                - The predicted body model parameters.
            - 'focal_length': float
        '''
        # GPU_monitor = GPUMonitor()
        x = self.normalize(x)
        B = len(x)

        # 1. Extract features from image.
        #  The input size is 256*256, but ViT needs 256*192. TODO: make this more elegant.
        feats = self.backbone(x[:, :, :, 32:-32])

        # 2. Run the head to predict the body model parameters.
        outputs = self.head(feats)  # pd_params [B,46], pd_cam [B,3]

        return outputs



    def build_cameras_kwargs(self, batch_size,focal_length):
        screen_size = torch.tensor([self.body_image_size, self.body_image_size]).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2).float(), 
            'focal_length': focal_length, 
            'image_size': screen_size, 'device': "cuda",
        }
        return cameras_kwargs
    
    def set_data_adaption(self, data_module_name=None):
        if data_module_name is None:
            # get_logger().warning('Data adapter schema is not defined. The input will be regarded as image patches.')
            self.adapt_batch = self._adapt_img_inference
        elif data_module_name == 'IMG_PATCHES':
            self.adapt_batch = self._adapt_img_inference
        elif data_module_name.startswith('SKEL_HSMR_V1'):
            self.adapt_batch = self._adapt_hsmr_v1
        else:
            raise ValueError(f'Unknown data module: {data_module_name}')

    def print_summary(self, max_depth=1):
        from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
        print(ModelSummary(self, max_depth=max_depth))

    def configure_optimizers(self):

        params_main = filter(lambda p: p.requires_grad, self._params_main())
        optimizer_main = torch.optim.AdamW( lr = 1e-05,  weight_decay =0.0001, params=params_main)


        return optimizer_main

    def _set_state(self,train=True):
        if train:
            self.backbone.train()
            self.head.train()
        else:
            self.backbone.eval()
            self.head.eval()




    def run_fit(self, init_iter=0):

        
        # build bar
        fit_bar = tqdm(range(init_iter, self._total_iters+1)) if self._debug else \
                  rtqdm(range(init_iter, self._total_iters+1))
        train_iter = iter(self.train_dataloader)
        self._set_state(train=True)
        writer = SummaryWriter(log_dir=self._writer_dir)
        os.makedirs(self._writer_dir, exist_ok=True)

        for iter_idx in fit_bar:
            # get data and prepare
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)


            # 1. Main parts forward pass.
            img_patch = to_tensor(batch['ehm_image'], self.lightning_fabric.device)  # (B, C, H, W)
            outputs = self.forward_step(img_patch)  # {...}     
            B = len(img_patch)


            pd_smplx_dict = self.ehm(outputs['body_param'], outputs['flame_param'],  pose_type='aa')

            pred_kps3d, mapping, _ = smplx_joints_to_dwpose(pd_smplx_dict['joints']) 
            pred_kps2d   = self.cameras.perspective_projection(pred_kps3d, \
                R = outputs['pd_cam'][:,:3,:3], T= outputs['pd_cam'][:,:3,3]) # [1,145,3]  55 + 68 + 22

            kps2d_mask =  batch['dwpose_kp2d'][:,:,2] > 0.7
            loss_dwpose_2d = self.metric( pred_kps2d[..., :2][kps2d_mask] / 1024,  batch['dwpose_kp2d'][..., :2][kps2d_mask] ) * 0.01 


            pred_smpl_3d = smplx2smpl_joints( pd_smplx_dict['vertices'],  self.smplx2smpl, self.smpl, self.J_regressor_extra, 'H36M-VAL-P2'  )  # [B, 44, 3]
            pred_smpl_2d   = self.cameras.perspective_projection(pred_smpl_3d, R = outputs['pd_cam'][:,:3,:3], T= outputs['pd_cam'][:,:3,3]) # [1,145,3]  55 + 68 + 22 

            loss_smpl_2d = self.keypoint_2d_loss(pred_smpl_2d[...,:2] / 1024,   batch['smpl_kp2d'] ) * 0.01   # 是不是可以加权
            loss_smpl_3d = self.keypoint_3d_loss(pred_smpl_3d,  batch['smpl_kp3d'], pelvis_id=39 ) * 0.05     

            # 3. Params Loss       pose shape .  body pose 需要加一个 confidence 
            loss_param_smplx = self.body_params_loss( outputs['body_param'],  batch['smplx_coeffs']  ) 
            loss_param_flame = self.head_params_loss( outputs['flame_param'], batch['flame_coeffs']  ) 

            loss_main =  loss_param_smplx + loss_param_flame + loss_smpl_3d + loss_smpl_2d + loss_dwpose_2d  # + loss_dwpose_3d


            if iter_idx % 50 ==0:
                writer.add_scalar("Loss/train_total", loss_main.item(), iter_idx)
                writer.add_scalar("Loss/param_smplx", loss_param_smplx.item(), iter_idx)
                writer.add_scalar("Loss/param_flame", loss_param_flame.item(), iter_idx)
                writer.add_scalar("Loss/loss_hmr_2d", loss_smpl_2d.item(), iter_idx)
                writer.add_scalar("Loss/loss_hmr_3d", loss_smpl_3d.item(), iter_idx)
                writer.add_scalar("Loss/dwpose_2d", loss_dwpose_2d.item(), iter_idx)


            # 5. Main parts backward pass.
            self.optimizer.zero_grad()  # self.backbone.blocks[0].mlp.fc1.weight.grad  
            self.lightning_fabric.backward(loss_main)
            self.optimizer.step()

            

            fit_bar.set_description(
            f"loss: {loss_main.item():.4f} | hmr 3D: {loss_smpl_3d.item():.3f} | hmr 2D: {(loss_smpl_2d.item()):.3f}  | 2D: {(loss_dwpose_2d.item()):.3f} | Params: {(loss_param_smplx.item()+loss_param_flame.item()):.3f}"
            )

            # # visualization
            batch_images = batch['image'] 
            if iter_idx % 1000 == 0:
                with torch.no_grad():
                    rank = dist.get_rank() 
                    n_imgs = len(batch_images) 
                    lights=PointLights(device=self.lightning_fabric.device, location=[[0.0, -1.0, -10.0]])
                    img_indices = np.linspace(0, n_imgs - 1, 1, dtype=int)  # 5
                    save_path = os.path.join(self._dump_dir ,"visual_train")
                    os.makedirs(save_path,exist_ok=True)
                    for im_idx in img_indices:
                        _img = (np.clip(batch_images[im_idx].clone().cpu().numpy(),0,1)*255).transpose(1,2,0).astype(np.uint8)
                        _img = cv2.resize(_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                        orig_img = _img

                        if  batch['smpl_kp'][im_idx]:
                            _t_lmk_dwp = (pred_smpl_2d[im_idx,:,:2]).detach().cpu().numpy()
                            _landmark_dwp = batch['smpl_kp2d'][im_idx, ...].detach().cpu().numpy()
                            _img = draw_landmarks(_landmark_dwp[:,:2] * 1024,  _img,  color=(0, 255, 0), viz_index = True ) # green
                            _img = draw_landmarks(_t_lmk_dwp,  _img,  color=(255, 0, 0),  viz_index = True ) # red
                        else:
                            _t_lmk_dwp = (pred_kps2d[im_idx,:,:2]).detach().cpu().numpy()
                            _landmark_dwp = batch['dwpose_kp2d'][im_idx, ...].detach().cpu().numpy()
                            _img = draw_landmarks(_landmark_dwp * 1024 ,  _img,  color=(0, 255, 0), viz_index = True ) # green
                            _img = draw_landmarks(_t_lmk_dwp,  _img,  color=(255, 0, 0),  viz_index = True ) # red
                        

                        pd_camera = GS_Camera(**self.build_cameras_kwargs(1,24), R = outputs['pd_cam'][im_idx:im_idx+1,:3,:3], T = outputs['pd_cam'][im_idx:im_idx+1,:3,3])
                        pd_mesh_img = self.body_renderer.render_mesh(pd_smplx_dict['vertices'][None, im_idx,...], pd_camera, lights=lights) 
                        pd_mesh_img = (pd_mesh_img[:,:3].cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                        pd_mesh_img = cv2.addWeighted( _img, 0.3, pd_mesh_img, 0.7, 0)

                        # batch['smplx_coeffs']['camera_RT_params']  =  batch['w2c_cam']
                        gt_smplx_dict = self.ehm(batch['smplx_coeffs'], batch['flame_coeffs'],  pose_type='aa')
                        gt_camera = GS_Camera(**self.build_cameras_kwargs(1,24), R = batch['smplx_coeffs']['camera_RT_params'][im_idx:im_idx+1,:3,:3], \
                                        T= batch['smplx_coeffs']['camera_RT_params'][im_idx:im_idx+1,:3,3]  )
                        gt_mesh_img = self.body_renderer.render_mesh( gt_smplx_dict['vertices'][None, im_idx,...], gt_camera, lights=lights ) 
                        gt_mesh_img = (gt_mesh_img[:,:3].cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                        gt_mesh_img = cv2.addWeighted( _img, 0.3, gt_mesh_img, 0.7, 0)

                        
                        _img = np.concatenate((orig_img, gt_mesh_img, pd_mesh_img), axis=1)
                        _img = cv2.resize(_img, ( _img.shape[1] // 2, _img.shape[0] // 2 ), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(save_path,f"smplx_stp_{iter_idx}_{im_idx}_{rank}.png"), cv2.cvtColor(_img.copy(), cv2.COLOR_RGB2BGR))
            
            with torch.no_grad():
                if iter_idx % 5000 == 0 and iter_idx != 0:
                    self.run_val(iter_idx, rank)

                if iter_idx % 40000 == 0 and iter_idx != 0:
                    self.save_checkpoints('ehm_model.pt',iter_idx)

        writer.close()
        
        del self.optimizer, self.train_dataloader
        torch.cuda.empty_cache()

    def save_checkpoints(self, name='latest.pt',iter_idx=1, optimizer=False):
        if self._debug:
            return
        saving_path = os.path.join(self._dump_dir, 'stage1_checkpoints')
        # remove old best model
        try:
            if name.startswith('best'):
                models = os.listdir(saving_path)
                for m in models:
                    if m.startswith('best'):
                        os.remove(os.path.join(saving_path, m))
        except:
            pass

        state = {'backbone': self.backbone, 'meta_cfg': self.cfg._dump, 'global_iter':iter_idx,
                 'head':self.head}
        if optimizer:
            state['optimizer'] = self.optimizer

        self.lightning_fabric.save(os.path.join(saving_path, name), state)
        print('Model saved at {}.'.format(os.path.join(saving_path, name)))


    def test_mesh (self, outputs, batch_images, im_idx=0, save_path="test_images"):
        lights=PointLights(device=self.lightning_fabric.device, location=[[0.0, -1.0, -10.0]])
        pd_smplx_dict = self.ehm(outputs['body_param'], outputs['flame_param'],  pose_type='aa')
        _img = (np.clip(batch_images[im_idx].clone().cpu().numpy(),0,1)*255).transpose(1,2,0).astype(np.uint8)
        _img = cv2.resize(_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        pd_camera = GS_Camera(**self.build_cameras_kwargs(1,24), R = outputs['pd_cam'][im_idx:im_idx+1,:3,:3], T = outputs['pd_cam'][im_idx:im_idx+1,:3,3])
        pd_mesh_img = self.body_renderer.render_mesh( pd_smplx_dict['vertices'][None, im_idx,...], pd_camera, lights=lights ) 
        pd_mesh_img = (pd_mesh_img[:,:3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
        pd_mesh_img = cv2.addWeighted( _img, 0.3, pd_mesh_img, 0.7, 0)
        cv2.imwrite(os.path.join(save_path, f"val_mesh_{im_idx}.png"), cv2.cvtColor(_img.copy(), cv2.COLOR_RGB2BGR))

    def run_val(self, cur_iter,rank):

        # val_iter = iter(self.val_dataloader
        val_iter = iter(self.val_dataloader)

        sample_batches = 2
        val_batches = []

        for i, batch in enumerate(val_iter):
            if i >= sample_batches:
                break
            val_batches.append(batch)
        iter_idx = 0

        for batch in val_batches:
            # 1. Main parts forward pass.
            # tmp_ori_pose = outputs['body_param']['global_pose'].clone()    tmp_body_pose = outputs['body_param']['body_pose'].clone()
            img_patch = to_tensor(batch['ehm_image'], self.lightning_fabric.device)  # (B, C, H, W)
            outputs = self.forward_step(img_patch)  # {...}     


            pd_smplx_dict = self.ehm(outputs['body_param'], outputs['flame_param'],  pose_type='aa')
            gt_smplx_dict = self.ehm(batch['smplx_coeffs'], batch['flame_coeffs'],  pose_type='aa')

            pd_proj_joints   = self.cameras.perspective_projection(pd_smplx_dict['joints'], \
                R = outputs['pd_cam'][:,:3,:3], T= outputs['pd_cam'][:,:3,3]) # [1,145,3]  55 + 68 + 22 
            pred_kps2d = smplx_joints_to_dwpose(pd_proj_joints)[0]  # [1,145,3] -> [1,134,3]

            
            # # visualization
            batch_images = batch['image']
        
            n_imgs = len(batch_images)
            lights=PointLights(device=self.lightning_fabric.device, location=[[0.0, -1.0, -10.0]])
            img_indices = np.linspace(0, n_imgs - 1, 1, dtype=int)  # 5
            save_path = os.path.join(self._dump_dir , "visual_val")
            os.makedirs(save_path,exist_ok=True)

            for im_idx in img_indices:
                _img = (np.clip(batch_images[im_idx].clone().cpu().numpy(),0,1)*255).transpose(1,2,0).astype(np.uint8)
                _img = cv2.resize(_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

                _t_lmk_dwp = pred_kps2d[im_idx,:,:2].detach().cpu().numpy()


                if  batch['smpl_kp'][im_idx]:
                    _t_lmk_dwp = (pred_kps2d[im_idx,:,:2]).detach().cpu().numpy()
                    _landmark_dwp = batch['smpl_kp2d'][im_idx, ...].detach().cpu().numpy()
                    _img = draw_landmarks(_landmark_dwp[:,:2] * 1024,  _img,  color=(0, 255, 0), viz_index = False ) # green
                    _img = draw_landmarks(_t_lmk_dwp,  _img,  color=(255, 0, 0),  viz_index = True ) # red
                else:
                    _t_lmk_dwp = (pred_kps2d[im_idx,:,:2]).detach().cpu().numpy()
                    _landmark_dwp = batch['dwpose_kp2d'][im_idx, ...].detach().cpu().numpy()
                    _img = draw_landmarks(_landmark_dwp * 1024 ,  _img,  color=(0, 255, 0), viz_index = False ) # green
                    _img = draw_landmarks(_t_lmk_dwp,  _img,  color=(255, 0, 0),  viz_index = True ) # red

           
                
                _img = draw_landmarks(_landmark_dwp,  _img,  color=(0, 255, 0), viz_index = True) # green
                _img = draw_landmarks(_t_lmk_dwp,  _img,  color=(255, 0, 0), viz_index = True ) # red

                gt_t_camera = GS_Camera(**self.build_cameras_kwargs(1,24), R = batch['smplx_coeffs']['camera_RT_params'][im_idx:im_idx+1,:3,:3], \
                                T= batch['smplx_coeffs']['camera_RT_params'][im_idx:im_idx+1,:3,3]  )
                gt_mesh_img = self.body_renderer.render_mesh( gt_smplx_dict['vertices'][None, im_idx,...], gt_t_camera, lights=lights ) 
                gt_mesh_img = (gt_mesh_img[:,:3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                gt_mesh_img = cv2.addWeighted( _img, 0.3, gt_mesh_img, 0.7, 0)

                pd_camera = GS_Camera(**self.build_cameras_kwargs(1,24), R = outputs['pd_cam'][im_idx:im_idx+1,:3,:3], T = outputs['pd_cam'][im_idx:im_idx+1,:3,3])
                pd_mesh_img = self.body_renderer.render_mesh(pd_smplx_dict['vertices'][None, im_idx,...], pd_camera, lights=lights) 
                pd_mesh_img = (pd_mesh_img[:,:3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                pd_mesh_img = cv2.addWeighted( _img, 0.3, pd_mesh_img, 0.7, 0)



                _img = np.concatenate((_img, gt_mesh_img, pd_mesh_img), axis=1)
                _img = cv2.resize(_img, ( _img.shape[1] // 2, _img.shape[0] // 2 ), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(save_path,f"val_stp_{cur_iter}_{iter_idx}_{rank}.png"), cv2.cvtColor(_img.copy(), cv2.COLOR_RGB2BGR))
                iter_idx += 1





    def transform_points_to_ndc(self, points, full_mat):

            
        N, P, _3 = points.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_h = torch.cat([points, ones], dim=2)
        points_ndc=torch.einsum('bij,bnj->bni',full_mat,points_h)
        
        points_ndc_xyz=points_ndc[:,:,:3]/(points_ndc[:,:,3:]+1e-7)
        points_ndc_xyz[:,:,2]=points_ndc[:,:,3] #  retain z range


        return points_ndc_xyz
    def transform_points_to_screen(self, points, full_mat, image_size = [256,256],  with_xyflip = True):

        
        points_ndc=self.transform_points_to_ndc(points,full_mat)
        
        N, P, _3 = points_ndc.shape

        if not torch.is_tensor(image_size):
            image_size = torch.tensor(image_size, device=self.device)
        if image_size.dim()==2:
            image_size = image_size[:,None]
        image_size=image_size[:,:,[1,0]]#width height
        
        points_screen=points_ndc.clone()
        points_screen[...,:2]=points_ndc[...,:2]*image_size/2-image_size/2
        if with_xyflip:
            points_screen[...,:2]=points_screen[:,:,:2]*-1
            
        return points_screen


    # ========== Internal Functions ==========

    def _params_main(self):
        return list(self.head.parameters()) + list(self.backbone.parameters())

    def _params_disc(self):
        if self.discriminator is None:
            return []
        else:
            return list(self.discriminator.parameters())

    @staticmethod
    def _adapt_skel_params(params):
        ''' Change the parameters formed like [pose_orient, pose_body, betas, trans] to [poses, betas, trans]. '''
        adapted_params = {}

        if 'poses' in params.keys():
            adapted_params['poses'] = params['poses']
        elif 'poses_orient' in params.keys() and 'poses_body' in params.keys():
            poses_orient = params['poses_orient']  # (B, 3)
            poses_body = params['poses_body']  # (B, 43)
            adapted_params['poses'] = torch.cat([poses_orient, poses_body], dim=1)  # (B, 46)
        else:
            raise ValueError(f'Cannot find the poses parameters among {list(params.keys())}.')

        if 'betas' in params.keys():
            adapted_params['betas'] = params['betas']  # (B, 10)
        else:
            raise ValueError(f'Cannot find the betas parameters among {list(params.keys())}.')

        return adapted_params


    def _init_backbone(self):
        print(f'Loading backbone weights from {self.cfg.BACKBONE.backbone_ckpt}')
        state_dict = torch.load(self.cfg.BACKBONE.backbone_ckpt, map_location='cpu')['state_dict']
        missing, unexpected = self.backbone.load_state_dict(state_dict)

    def compute_losses_main(
        self,
        outputs,
        batch
    ) -> Tuple[torch.Tensor, Dict]:
        ''' Compute the weighted losses according to the config file. '''

        # 1. Preparation.
        B = len(outputs['pd_cam'])


        # 2. Keypoints losses.
        pd_smplx_dict = self.ehm(outputs['body_param'], outputs['flame_param'],  pose_type='aa')
        pd_proj_joints   = self.cameras.perspective_projection(pd_smplx_dict['joints'], \
            R = outputs['pd_cam'][:,:3,:3], T= outputs['pd_cam'][:,:3,3]) # [1,145,3]  55 + 68 + 22 
        pred_kps2d, mapping, _ = smplx_joints_to_dwpose(pd_proj_joints)  # [1,145,3] -> [1,134,3]

        # TODO:               在这里加几个loss，      batch['dwpose_rlt']['kp3d']  batch['dwpose_rlt']['kp2d'] 
        kps2d_mask =  batch['dwpose_rlt']['scores'] > 0.7 
        # loss_3d = self.metric(pd_joints[:, mapping][kps2d_mask].unsqueeze(dim=1), gt_joints[:, mapping][kps2d_mask].unsqueeze(dim=1)) * 0.01   # 这玩意是不是也得加一个 confidence
        loss_dwpose_2d = self.metric( pred_kps2d[..., :2][kps2d_mask] / 1024, batch['dwpose_rlt']['keypoints'][kps2d_mask] ) / B   #  有 bug


        pred_3d = smplx2smpl_joints(pd_smplx_dict['vertices'],  self.smplx2smpl, self.smpl, self.J_regressor_extra  )  # [B, 44, 3]
        pred_2d   = self.cameras.perspective_projection(pred_3d, \
                R = outputs['pd_cam'][:,:3,:3], T= outputs['pd_cam'][:,:3,3]) # [1,145,3]  55 + 68 + 22 

        kp2d_loss = self.keypoint_2d_loss(pred_2d[...,:2] / 1024,   batch['dwpose_rlt']['kp2d'] ) / B   # 是不是可以加权
        kp3d_loss = self.keypoint_3d_loss(pred_3d,  batch['dwpose_rlt']['kp3d'] , pelvis_id=39 ) / B 



        # 3. Params Loss       pose shape .  body pose 需要加一个 confidence
        loss_param_smplx = self.body_params_loss( outputs['body_param'], batch['smplx_coeffs']  ) 
        loss_param_flame = self.head_params_loss( outputs['flame_param'], batch['flame_coeffs']  ) 

        gt_flame_param = batch['flame_coeffs']
        param2 = torch.cat([gt_flame_param['eye_pose_params'], 
        gt_flame_param['jaw_params'],gt_flame_param['eyelid_params'],gt_flame_param['expression_params'],gt_flame_param['shape_params']] ,dim = 1)



    def _adapt_img_inference(self, img_patches):
        return {'img_patch': img_patches}