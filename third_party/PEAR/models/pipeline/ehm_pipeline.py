
from models.backbones import ViT
from models.smplx.smplx_head import SMPLXTransformerDecoderHead
import torch
from utils.pipeline_utils import *
import os
from datetime import datetime
import lightning as L
import torch,torchvision
import numpy as np
from torchvision.utils import save_image

class Ehm_Pipeline(L.LightningModule):
    def __init__(self, cfg):
        super(Ehm_Pipeline, self).__init__()
        self.cfg = cfg
        self._dump_dir =  os.path.join('outputs', "test", datetime.now().strftime("%Y%m%d_%H"))
        self._total_iters = cfg.TRAIN.train_iter
        self._check_interval = cfg.TRAIN.check_interval 
        self._visual_train_interval = 1000 
        self._debug = False
        self.body_image_size = 1024
        self.head_image_size = 512
        self.backbone = ViT(**cfg.BACKBONE)
        self.head = SMPLXTransformerDecoderHead(cfg.HEAD, cfg.TRAIN.batch_size)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def forward(self, x:torch.Tensor):
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
        # save_image(x[:, :, :, 32:-32],"input.jpg")
        x = self.normalize(x)
        feats = self.backbone(x[:, :, :, 32:-32])

        # 2. Run the head to predict the body model parameters.
        outputs = self.head(feats)  # pd_params [B,46], pd_cam [B,3]

        return outputs

