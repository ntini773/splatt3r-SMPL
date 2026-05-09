import os
import json
import torch
import pickle
import random
import numpy as np
import torchvision
from copy import deepcopy
import time

from torch.utils.data import DataLoader
import pycocotools.mask as mask_utils

from utils.lmdb import LMDBEngine
from utils.graphics_utils import get_full_proj_matrix, show_image
import torch.utils.data._utils.collate as collate
import torch
import matplotlib.pyplot as plt
import imageio
import webdataset as wds
import torch
import io
from PIL import Image
from typing import List, Union
import braceexpand
from torchvision import transforms
import torch
import numpy as np
from skimage.transform import rotate, resize
from skimage.filters import gaussian
import random
import cv2
from typing import List, Dict, Tuple
from yacs.config import CfgNode
from .dataset_utils import get_example, expand_to_aspect_ratio

class MixedWebDataset(wds.WebDataset):
    def __init__(self) -> None:
        super(wds.WebDataset, self).__init__()


def pt_decoder(key, value):
    if key.endswith(".pt"):
        return {"data": torch.load(io.BytesIO(value), map_location='cpu')}
    else:
        return {key: value}


DEFAULT_MEAN = 255. * np.array([0., 0., 0.]) # np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([1., 1., 1.])
DEFAULT_IMG_SIZE = 256  

body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]  # SMPL 的 flip 



body_permutation_dwpose = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16, 21, 22, 23, 18, 19, 20] # 24
face_permutation_dwpose = [i for i in range(24, 92)]  #  68
hand_permutation_dwpose = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
                            92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]  # 42

FLIP_KEYPOINT_PERMUTATION_DWPOSE = body_permutation_dwpose + face_permutation_dwpose + hand_permutation_dwpose  





to_tensor = transforms.ToTensor() 

def decode_images(sample):
    """将 bytes 图像转为 torch.Tensor，范围 [0,1]"""
    img = Image.open(io.BytesIO(sample["body_image"])).convert("RGB")
    sample["body_image"] = img   # [3, H, W], float32, [0,1]
    if sample["body_mask"] is not None:
        mask = Image.open(io.BytesIO(sample["body_mask"])).convert("L")  
        sample["body_mask"] =  mask   # [1, H, W], float32, [0,1]

    return sample

def expand_urls(urls: Union[str, List[str]]):

    def expand_url(s):
        return os.path.expanduser(os.path.expandvars(s))

    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand_url(url))]
    return urls




def data_to_tensor(data_dict, device='cpu'):
    assert isinstance(data_dict, dict), 'Data must be a dictionary.'
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray) or isinstance(data_dict[key], list):
            data_dict[key] = torch.as_tensor(data_dict[key], device=device,dtype=torch.float32)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = data_to_tensor(data_dict[key], device=device)
        else:
            continue
    return data_dict

def squeeze_params(tracking_info):
    tracking_info['smplx_coeffs']  = {kk: vv.squeeze() for kk, vv in tracking_info['smplx_coeffs'].items()}
    # tracking_info['left_mano_coeffs'] = {kk: vv.squeeze() for kk, vv in tracking_info['left_mano_coeffs'].items()}
    # tracking_info['right_mano_coeffs'] = {kk: vv.squeeze() for kk, vv in tracking_info['right_mano_coeffs'].items()}
    # tracking_info['flame_coeffs']  = {kk: vv.squeeze() for kk, vv in tracking_info['flame_coeffs'].items()}
    return tracking_info



    # tracking_info['head_box'],tracking_info['left_hand_box'],tracking_info['right_hand_box'] = self._load_box(tracking_info)


def apply_example_formatter(dataset):


    def fet_tracking_info_from_raw(sample):
        
        if  sample['id_params']['smplx_shape'].shape[0] == 10:
            smplx_shape = np.concatenate([sample['id_params']['smplx_shape'], np.zeros((200-len(sample['id_params']['smplx_shape'])),np.float32)])
            flame_shape = np.concatenate([sample['id_params']['flame_shape'], np.zeros((300-len(sample['id_params']['flame_shape'])),np.float32)])
        else:
            smplx_shape = sample['id_params']['smplx_shape'][0]
            flame_shape = sample['id_params']['flame_shape'][0]

        if sample['id_params']['joints_offset'][0][0][0] == 1:  # for sam3d bug case
            sample['id_params']['joints_offset'] *=  0


        tracking_info = {}
        tracking_info['smplx_coeffs'] = sample['smplx_params']  # exp 50 
        tracking_info['smplx_coeffs'].update({"shape":smplx_shape,  # 10  200 dim
                                                "joints_offset":sample['id_params']['joints_offset'],
                                                "head_scale":sample['id_params']['head_scale'],
                                                "hand_scale":sample['id_params']['hand_scale'],
                                                })
        
        tracking_info['flame_coeffs'] = sample['flame_params']
        tracking_info['flame_coeffs'].update({"shape_params":flame_shape})  # 10  300



        hmr_pose = tracking_info['smplx_coeffs'].get("hmr_pose") # just body pose,
        if hmr_pose is not None:  
            tracking_info['smplx_coeffs']["body_pose"][[0,1,3,4,6,7,9,10]] = hmr_pose[[0,1,3,4,6,7,9,10]] 
        else:
            tracking_info['smplx_coeffs']["hmr_pose"] = tracking_info['smplx_coeffs']["body_pose"] 


        tracking_info=data_to_tensor(tracking_info)  
        tracking_info=squeeze_params(tracking_info)

        # tracking_info=data_to_tensor(tracking_info)
        # tracking_info=squeeze_params(tracking_info)
        
        # Convert PyTorch 3D coordinate system to the COLMAP coordinate system. 
        # (Since it is identical to the image coordinate, the same camera parameters 
        tracking_info['flame_coeffs']['has_flame'] =  int(sample['head_valid']) 
        tracking_info['smplx_coeffs']['has_hand']  =  int(sample['hand_valid'])
        tracking_info['smplx_coeffs']['has_body']  =  int(sample['pose_valid']) 

        # are employed for unprojection and Gaussian rendering.)
        RT = tracking_info['smplx_coeffs']['camera_RT_params']
        c2c_mat=torch.tensor([[-1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 1, 0],
                                [ 0, 0, 0, 1],
                                ],dtype=torch.float32)
        RT_mat=torch.tensor([[  1, 0, 0, 0],
                                [ 0, 1, 0, 0],
                                [ 0, 0, 1, 0],
                                [ 0, 0, 0, 1],
                                ],dtype=torch.float32)
        RT_mat[:3,:4]=RT
        w2c_cam=torch.matmul(c2c_mat,RT_mat)
        c2w_cam=torch.linalg.inv(w2c_cam)
        tracking_info['w2c_cam'],tracking_info['c2w_cam']=w2c_cam,c2w_cam
        return tracking_info

    def example_formatter(sample):


        random_idx = random.randint(0,len(sample['annotation.pyd'])-1)
        cur_annotation = sample['annotation.pyd'][random_idx]

        source_tracking_info = fet_tracking_info_from_raw(cur_annotation)  


        if cur_annotation.get('smpl_keypoints_2d') is not None:
            kp2d = cur_annotation.get('smpl_keypoints_2d')
            kp3d = cur_annotation.get('smpl_keypoints_3d')
            source_tracking_info['smpl_kp'] = True
            flip_kp = FLIP_KEYPOINT_PERMUTATION
        else:
            kp2d = cur_annotation.get('dwpose_keypoints_2d')
            kp3d = cur_annotation.get('dwpose_keypoints_3d')
            source_tracking_info['smpl_kp'] = False
            flip_kp = FLIP_KEYPOINT_PERMUTATION_DWPOSE



        # left hand     right hand     hand scale     hmr pose valid   
        smpl_params = {'global_orient': source_tracking_info['smplx_coeffs']['global_pose'].numpy(),
                    'body_pose': source_tracking_info['smplx_coeffs']['body_pose'].reshape(-1).numpy(),
                    'left_hand_pose': source_tracking_info['smplx_coeffs']['left_hand_pose'].reshape(-1).numpy(),
                    'right_hand_pose': source_tracking_info['smplx_coeffs']['right_hand_pose'].reshape(-1).numpy(),
                    'betas': source_tracking_info['smplx_coeffs']['shape'].numpy(),
                    'has_flame':  source_tracking_info['flame_coeffs']['has_flame'],  
                    }

        bbox_size = cur_annotation['scale'].max()
        center_x = cur_annotation['center'][0]  # int
        center_y = cur_annotation['center'][1]  # 


        original_image =  sample["jpg"] 

        if "mask" in cur_annotation:
            m = cur_annotation["mask"]
            source_mask = mask_utils.decode(m)  
            source_tracking_info['render_valid'] = 1.0
        else:
            source_mask = np.ones((original_image.shape[0], original_image.shape[1]), dtype=np.uint8) # 全 1 mask
            source_tracking_info['render_valid'] = 0.0


        img_rgba = np.concatenate([original_image, source_mask[:,:,None]], axis=2).astype(np.uint8)
        img_patch_rgba, keypoints_2d, keypoints_3d, smpl_params, img_size, trans = get_example(img_rgba,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    kp2d, kp3d,
                                                                                                    smpl_params,
                                                                                                    flip_kp,  # bug 12.16 todo
                                                                                                    256, 256,
                                                                                                    DEFAULT_MEAN, DEFAULT_STD, 
                                                                                                    do_augment = True, 
                                                                                                    is_bgr=False, return_trans = True,
                                                                                                    use_skimage_antialias=False,
                                                                                                    border_mode=0, # 0
                                                                                                    )


        source_tracking_info['smplx_coeffs']['global_pose'] = torch.tensor(smpl_params['global_orient'])
        source_tracking_info['smplx_coeffs']['body_pose'] = torch.tensor(smpl_params['body_pose'].reshape(-1,3))
        source_tracking_info['smplx_coeffs']['shape'] = torch.tensor( smpl_params['betas'] )
        
        source_tracking_info['smplx_coeffs']['left_hand_pose'] = torch.tensor(smpl_params['left_hand_pose'].reshape(-1,3))
        source_tracking_info['smplx_coeffs']['right_hand_pose'] = torch.tensor(smpl_params['right_hand_pose'].reshape(-1,3))

        source_tracking_info['flame_coeffs']['has_flame'] += 0.1
        source_tracking_info['smplx_coeffs']['has_body'] += 0.1



        img_patch = torch.tensor(img_patch_rgba[:3,:,:],dtype=torch.float32)
        mask_patch = torch.tensor((img_patch_rgba[3:,:,:] ).clip(0,1))


        if source_tracking_info['smpl_kp']:
            source_tracking_info['smpl_kp3d'] = torch.tensor(keypoints_3d )  # [44,4]
            source_tracking_info['smpl_kp2d'] = torch.tensor(keypoints_2d )  # [44,3]
            source_tracking_info['dwpose_kp3d'] = torch.zeros((134,4) )  # [44,4]
            source_tracking_info['dwpose_kp2d'] = torch.zeros((134,3) )  # [44,3]
        else:
            source_tracking_info['smpl_kp3d'] =  torch.zeros((44,4) )  # [44,4]
            source_tracking_info['smpl_kp2d'] =  torch.zeros((44,3) )  # [44,3]
            source_tracking_info['dwpose_kp3d'] = torch.tensor(keypoints_3d )  # [44,4]
            source_tracking_info['dwpose_kp2d'] = torch.tensor(keypoints_2d )  # [44,3]

            source_tracking_info['dwpose_kp2d'][8] = source_tracking_info['dwpose_kp2d'][8] * 0  # 
            source_tracking_info['dwpose_kp2d'][11] = source_tracking_info['dwpose_kp2d'][11] * 0 


        img_patch = torch.tensor(img_patch_rgba[:3,:,:],dtype=torch.float32)
        mask_patch = torch.tensor((img_patch_rgba[3:,:,:] ).clip(0,1))



        source_image = img_patch  # * source_mask  + (1-source_mask) * self.bg_color  
        ehm_source_image = source_image #  torchvision.transforms.functional.resize(source_image, (256, 256), antialias=True)
        source_image = torchvision.transforms.functional.resize(source_image, (518, 518), antialias=True) # resize [3,518,518]
        source_mask = torchvision.transforms.functional.resize(mask_patch, (518, 518), antialias=True)

        target_image = torchvision.transforms.functional.resize(img_patch, (512, 512), antialias=True) # resize [3,518,518]
        target_mask = torchvision.transforms.functional.resize(source_mask, (512, 512), antialias=True)

        view_matrix,  full_proj_matrix  =  get_full_proj_matrix(source_tracking_info['w2c_cam'],  1/24)

        source_tracking_info['render_cam_params']={  # target info 多一个这个 和  mask.  目标渲染视角 
            "world_view_transform":view_matrix,"full_proj_transform":full_proj_matrix,
            'tanfovx': 1/24,     'tanfovy': 1/24,
            'image_height':512,  'image_width':512,
            'camera_center':source_tracking_info['c2w_cam'][:3,3]
        }
        



        source_tracking_info['image'] = source_image         # [3,518,518] 518 resolution for ehm
        source_tracking_info['mask'] = source_mask           # [1,518,518]
        source_tracking_info['ehm_image'] = ehm_source_image # [3,256,256]  256 resolution for ehm
        source_tracking_info['target_image'] = target_image  # 512 resolution for render supervisition
        source_tracking_info['target_mask'] = target_mask
        source_tracking_info['image_name'] = sample['__key__']


        return source_tracking_info
    


    dataset = dataset.map(example_formatter)
    return dataset


def load_tars_as_wds( urls,  resampled,  split):

    urls = expand_urls(urls)  # to list of URL strings
    if split == 'test':
        dataset : wds.WebDataset = wds.WebDataset(
                urls, # 所有数据位置
                nodesplitter = wds.split_by_node,  # 分布式环境下的数据分片器
                shardshuffle = False,  # 是否对 .tar 文件顺序打乱
            )
    

    else:
        dataset : wds.WebDataset = wds.WebDataset(
                urls, # 所有数据位置
                nodesplitter = wds.split_by_node,  # 分布式环境下的数据分片器
                shardshuffle = True,  # 是否对 .tar 文件顺序打乱
                resampled    = True,  # resampled, # 每个 epoch 从数据集中采样 resampled 个样本，流式
                cache_dir    = None,  # 	是否启用缓存，默认关闭
            )

    dataset = dataset.decode("rgb8")
    dataset = dataset.rename(jpg="jpg;jpeg;png")
    dataset = apply_example_formatter(dataset)

    return dataset


def build_test_web_tracked_data(cfg_dataset, split, val_name = None, shuffle=True):
    names, datasets, weights = [], [], []
    if split == 'train':
        dataset_type = cfg_dataset.datasets
        Train = True
    elif split == 'valid' or split == 'test':
        dataset_type = cfg_dataset.val_datasets
        Train = False  # For debug

        
    for ds_cfg in dataset_type:  # Iterate over all datasets in the training configuration.
        if Train == False and ds_cfg['name'] != val_name:
            continue
        dataset = load_tars_as_wds(
                ds_cfg.item.urls,
                ds_cfg.item.epoch_size,
                split
        )

        names.append(ds_cfg.name)
        datasets.append(dataset)
        weights.append(ds_cfg.weight) # 权重
        # weights.append(0.1)
        # get_logger().info(f"Dataset '{ds_cfg.name}' loaded.")

    # Normalize the weights and mix the datasets.
    weights = np.array(weights)
    weights = weights / weights.sum()
    train_dataset = MixedWebDataset()
    train_dataset.append(wds.RandomMix(datasets, weights)) 


    if split =='test':
        return train_dataset

    if Train:
        train_dataset = train_dataset.with_epoch(50_000).shuffle(1000, initial=1000)
    else:
        if shuffle:
            train_dataset = train_dataset.with_epoch(1000).shuffle(1000, initial=1000)
        else:
            train_dataset = train_dataset.with_epoch(1000)

    return train_dataset


def build_web_tracked_data(cfg_dataset, split, shuffle=True):
    names, datasets, weights = [], [], []
    if split == 'train':
        dataset_type = cfg_dataset.datasets
        Train = True
    elif split == 'valid' or split == 'test':
        dataset_type = cfg_dataset.val_datasets
        Train = False  # For debug

        
    for ds_cfg in dataset_type:  # Iterate over all datasets in the training configuration.
        dataset = load_tars_as_wds(
                ds_cfg.item.urls,
                ds_cfg.item.epoch_size,
                split
        )

        names.append(ds_cfg.name)
        datasets.append(dataset)
        weights.append(ds_cfg.weight) # 权重
        # weights.append(0.1)
        # get_logger().info(f"Dataset '{ds_cfg.name}' loaded.")

    # Normalize the weights and mix the datasets.
    weights = np.array(weights)
    weights = weights / weights.sum()
    train_dataset = MixedWebDataset()
    train_dataset.append(wds.RandomMix(datasets, weights)) 


    if split =='test':  
        return train_dataset  
    
 

    if Train:
        train_dataset = train_dataset.with_epoch(50_000).shuffle(1000, initial=1000)
    else:
        if shuffle:
            train_dataset = train_dataset.with_epoch(1000).shuffle(1000, initial=1000)
        else:
            train_dataset = train_dataset.with_epoch(1000)

    return train_dataset
