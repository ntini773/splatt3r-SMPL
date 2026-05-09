import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse

from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import (
    ConfigDict
)
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from pytorch3d.renderer import PointLights
import os
from models.modules.ehm import EHM_v2 

from utils.graphics_utils import GS_Camera
import cv2
import torchvision.transforms.functional as TF
from smplx import SMPL, SMPLX
import joblib
import pickle
from smplx.lbs import vertices2joints
import torch
import numpy as np
import pickle
from typing import Optional
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput
import rich
import torchvision
DEFAULT_HSMR_ROOT = 'configs/eval'
import torch
from typing import Any, Dict, List
import trimesh



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smplx2smpl = torch.from_numpy(joblib.load("assets/SMPLX2SMPL/body_models/smplx2smpl.pkl")['matrix']).unsqueeze(0).float().cuda()
smpl = SMPL("assets/SMPL/SMPL_NEUTRAL.pkl", gender='neutral').to(device)

J_regressor_extra = torch.tensor(pickle.load( open("assets/SMPLX2SMPL/SMPL_to_J19.pkl", 'rb'), # SMPL_to_J19.pkl
                    encoding='latin1'), dtype=torch.float32).to(device) 


DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406])
DEFAULT_STD =  torch.tensor([0.229, 0.224, 0.225])
def number_to_rgb(n):
    n = int(n)
    r = (n >> 16) & 0xFF
    g = (n >> 8) & 0xFF
    b = n & 0xFF
    return (r, g, b)

smplx_to_openpose =[
    134,  # 鼻子 68
    12,  # 脖子
    17,19,21, # 右手 
    16,18,20, # 左手 
    0, # 骨盆
    2,5,8, #  右腿
    1,4,7, #  左腿
    24,23, #  眼睛
    136,125, # 耳朵
    124,132,127,  #  右脚 脚后跟的点好像不对
    135,143,138   # 左脚
]




smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34] # 因为数据集中只有这些点

smpl_to_h36m = [
    0, 1, 33, 3, 4, 34, 6, 7, 39, 27, 26, 11, 28, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 11, 26, 27, 28,  13, 
    14, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
]

smpl_to_coco = [
    0, 1, 33, 3, 4, 34, 6, 7, 39, 27, 10, 11, 28, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 11, 26, 27, 28,  13, 
    14, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
]

JOINT_MAP_H36M = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 49,
'OP RHip': 45, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 46, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose', # 0
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder', # 5
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee', # 10
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',  # 15 
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe', # 20
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',  # 24
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle', # 25   
'Right Knee',
'Right Hip', # dwpose
'Left Hip',  # dwpose
'Left Knee',
'Left Ankle', # 30
'Right Wrist',
'Right Elbow',
'Right Shoulder', # dwpose
'Left Shoulder',  # dwpose
'Left Elbow',  # 35
'Left Wrist',  
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',  # 40
'Spine (H36M)',   # 
'Jaw (H36M)',
'Head (H36M)',

# 后面五个不需要
# 'Nose',
# 'Left Eye',
# 'Right Eye',
# 'Left Ear',
# 'Right Ear'
]

JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}






def smplx2smpl_joints(smplx_vertics, smplx2smpl, smpl, J_regressor_extra , dataname ):
    batch = smplx_vertics.shape[0]
    smplx2smpl_expanded = smplx2smpl.expand(batch, -1, -1)  # [300, 6890, 10475]
    smpl_verts = torch.matmul(smplx2smpl_expanded, smplx_vertics[:, :-120])  # [300, 6890, 3]

    #### For COCO  and LSP
    if dataname in  ['LSP-EXTENDED',  'COCO-VAL']:
        J_regressor_extra = torch.tensor(np.load("assets/SMPLX2SMPL/data/J_regressor_extra.npy") ).float().to(device)# [9,6890]
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        joint_map = torch.tensor(joints, dtype=torch.long)
        extra_joints = vertices2joints(J_regressor_extra, smpl_verts)
        J_regressor_eval = smpl.J_regressor.clone().expand(batch, -1, -1) 
        target_j3d = torch.matmul(J_regressor_eval, smpl_verts) # [24, 3] 
        target_j3d = smpl.vertex_joint_selector(smpl_verts, target_j3d) # [45,3]
        joints = torch.cat([target_j3d, extra_joints], dim=1)
        joints = joints[:, joint_map, :]
        return joints


    if dataname in  ['H36M-VAL-P2', 'POSETRACK-VAL',  '3DPW-TEST' ]:
        J_regressor_eval = smpl.J_regressor.clone().expand(batch, -1, -1) 
        target_j3d = torch.matmul(J_regressor_eval, smpl_verts) # [24, 3] 
        target_j3d = smpl.vertex_joint_selector(smpl_verts, target_j3d) # [45,3]
        target_j3d = target_j3d[:, smpl_to_openpose, :] 
        extra_joints = vertices2joints(J_regressor_extra, smpl_verts)  # [B, 19, 3] 
    
        all_joints = torch.cat([target_j3d, extra_joints], dim=1)  # [B, 44, 3]
        all_joints = all_joints[:, smpl_to_h36m, :] # for h36m  evaluation 
        return all_joints  # [44, 3]  

