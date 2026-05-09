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
import copy
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
DEFAULT_IMG_SIZE = 256  # 这个可以设置大一些

body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]


# body_permutation_dwpose = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
# extra_permutation_dwpose = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]

# FLIP_KEYPOINT_PERMUTATION_DWPOSE = body_permutation + [25 + i for i in extra_permutation]





to_tensor = transforms.ToTensor()  # 转换为 [C, H, W] 的 float32 tensor，范围 [0,1]

def decode_images(sample):
    """将 bytes 图像转为 torch.Tensor，范围 [0,1]"""
    img = Image.open(io.BytesIO(sample["body_image"])).convert("RGB")
    sample["body_image"] = img   # [3, H, W], float32, [0,1]
    if sample["body_mask"] is not None:
        mask = Image.open(io.BytesIO(sample["body_mask"])).convert("L")  # 如果 mask 是灰度图
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
        
        

        tracking_info =  deepcopy(sample['params'])
        tracking_info['smplx_coeffs'].update({"shape":sample['id_params']['smplx_shape'][0],  # 加入一些共有的信息
                                                "joints_offset":sample['id_params']['joints_offset'][0],
                                                "head_scale":sample['id_params']['head_scale'][0],
                                                "hand_scale":sample['id_params']['hand_scale'][0],
                                                })
        hmr_pose = tracking_info['smplx_coeffs'].get("hmr_pose")

        if hmr_pose is not None:  # 是不是 bone joint 也能拿来训练
            tracking_info['smplx_coeffs']["body_pose"][[0,1,3,4,6,7,9,10]] = hmr_pose[[0,1,3,4,6,7,9,10]]  # 似乎脚踝那个可以不用 hmr 作为监督
        else:
            tracking_info['smplx_coeffs']["hmr_pose"] = tracking_info['smplx_coeffs']["body_pose"]

        tracking_info['flame_coeffs'].update({"shape_params":sample['id_params']['flame_shape'][0]})

        tracking_info=data_to_tensor(tracking_info)  
        tracking_info=squeeze_params(tracking_info)

        # tracking_info=data_to_tensor(tracking_info)
        # tracking_info=squeeze_params(tracking_info)
        
        # Convert PyTorch 3D coordinate system to the COLMAP coordinate system. 
        # (Since it is identical to the image coordinate, the same camera parameters 
        tracking_info['flame_coeffs']['has_flame'] = float(torch.mean(tracking_info['dwpose_rlt']['scores'][24: 92]) > 0.6)
        tracking_info['smplx_coeffs']['has_hand'] = float(torch.mean(tracking_info['dwpose_rlt']['scores'][92:]) > 0.6)


        # tracking_info['head_box'],tracking_info['left_hand_box'],tracking_info['right_hand_box'] = load_box(tracking_info)

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



    # 可以在这里检测一下是什么数据，  并且取消
    # 
    def example_formatter_hmr(sample):
        sample = decode_images(sample)
        orig_source_tracking_info = fet_tracking_info_from_raw(sample) 
        source_tracking_info = copy.deepcopy(orig_source_tracking_info)

        image,  mask = np.array(sample['body_image']) , np.array(sample['body_mask']) 

        hmr_params = sample['hmr_params']

        keypoints_2d = hmr_params['2dkp'].copy()  # np [44,3]
        keypoints_3d = hmr_params['3dkp'].copy() # np [44,4]
        center_x = hmr_params['center'][0]  # int
        center_y = hmr_params['center'][1]  # 

        scale = hmr_params['scale'] #  float
        pose_valid = hmr_params['pose_valid'] # array(1)
        beta_valid = hmr_params['beta_valid'] # array(1)



        bbox_size = expand_to_aspect_ratio(scale*200).max()
        if bbox_size < 1:
            breakpoint()
        # left hand     right hand     hand scale     hmr pose valid   
        smpl_params = {'global_orient': source_tracking_info['smplx_coeffs']['global_pose'].numpy(),
                    'body_pose': source_tracking_info['smplx_coeffs']['body_pose'].reshape(-1).numpy(),
                    'betas': source_tracking_info['smplx_coeffs']['shape'].numpy(),
                    }

        has_smpl_params = {'global_orient': pose_valid,
                        'body_pose': pose_valid,
                        'betas': beta_valid
                        }


        img_rgba = np.concatenate([image, mask[:,:,None]], axis=2).astype(np.uint8)
        img_patch_rgba, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size, trans = get_example(img_rgba,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    FLIP_KEYPOINT_PERMUTATION,
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

        source_tracking_info['smplx_coeffs']['has_body'] = pose_valid.item()
        source_tracking_info['smplx_coeffs']['has_beta'] = beta_valid.item()

        source_tracking_info['flame_coeffs']['has_flame'] = 0.  # 可以去掉
        source_tracking_info['smplx_coeffs']['has_hand'] = 0.


        source_tracking_info['dwpose_rlt']['kp3d'] = torch.tensor(keypoints_3d ) # [44,4]
        source_tracking_info['dwpose_rlt']['kp2d'] = torch.tensor(keypoints_2d  ) # [44,3]

        source_tracking_info['render_valid'] = 0.

        img_patch = torch.tensor(img_patch_rgba[:3,:,:],dtype=torch.float32)
        mask_patch = torch.tensor((img_patch_rgba[3:,:,:] ).clip(0,1))


        source_image = img_patch  # * source_mask  + (1-source_mask) * self.bg_color  再将人体抠出来  似乎没必要把人体抠出来
        ehm_source_image = source_image #  torchvision.transforms.functional.resize(source_image, (256, 256), antialias=True)
        source_image = torchvision.transforms.functional.resize(source_image, (518, 518), antialias=True) # resize [3,518,518]
        source_mask = torchvision.transforms.functional.resize(mask_patch, (518, 518), antialias=True)

        target_ehm = torchvision.transforms.functional.resize(img_patch, (256, 256), antialias=True) # resize [3,518,518]
        target_image = torchvision.transforms.functional.resize(img_patch, (512, 512), antialias=True) # resize [3,518,518]
        target_mask = torchvision.transforms.functional.resize(source_mask, (512, 512), antialias=True)

        view_matrix,  full_proj_matrix  =  get_full_proj_matrix(source_tracking_info['w2c_cam'],  1/24)

        source_tracking_info['render_cam_params'] = {  # target info 多一个这个 和  mask.  目标渲染视角 
            "world_view_transform":view_matrix,"full_proj_transform":full_proj_matrix,
            'tanfovx': 1/24,     'tanfovy': 1/24,
            'image_height':512,  'image_width':512,
            'camera_center':source_tracking_info['c2w_cam'][:3,3]
        }


        source_tracking_info['image'] = source_image         # [3,518,518] 518 resolution for ehm
        source_tracking_info['mask'] = source_mask           # [1,518,518]
        source_tracking_info['source_ehm'] = ehm_source_image # [3,256,256]  256 resolution for ehm
        source_tracking_info['target_ehm'] = target_ehm  # 512 resolution for render supervisition
        source_tracking_info['target_image'] = target_image  # 512 resolution for render supervisition
        source_tracking_info['target_mask'] = target_mask
        source_tracking_info['image_name'] = sample['__key__']
        source_tracking_info['source_info'] = orig_source_tracking_info

        return source_tracking_info



    def example_formatter_all(sample):


        frames = len(sample['frame_id_list'])
        source_idx, target_idx = random.sample(range(frames), 2)
        all_images, all_masks , all_params = sample["all_body_image" ], sample["all_body_mask" ],sample["all_params" ]

        source_sample =   {
            "params": all_params[source_idx],
            "id_params": sample[ "id_params"]
        }
        target_sample =   {
            "params": all_params[target_idx],
            "id_params": sample[ "id_params"]
        }

        source_tracking_info = fet_tracking_info_from_raw(source_sample) 
        target_tracking_info = fet_tracking_info_from_raw(target_sample) 

        image,  mask = to_tensor(Image.open(io.BytesIO(all_images[source_idx])).convert("RGB"))   , \
                       to_tensor(Image.open(io.BytesIO(all_masks[source_idx])).convert("L") )

        tagt_image,  tagt_mask = to_tensor(Image.open(io.BytesIO(all_images[target_idx])).convert("RGB"))   , \
                       to_tensor(Image.open(io.BytesIO(all_masks[target_idx])).convert("L") )



        params = target_sample['params']
        keypoints_2d = np.concatenate(
            [params['dwpose_rlt']['keypoints'], 
            np.ones((params['dwpose_rlt']['keypoints'].shape[0], 1))],
            axis=1
        )  # [44, 3]



        # 要不把这个设置为 0 试试？
        target_tracking_info['smplx_coeffs']['has_body'] = 1.
        target_tracking_info['smplx_coeffs']['has_beta'] = 1.

        target_tracking_info['render_valid'] = 1.

        target_tracking_info['dwpose_rlt']['keypoints'] = torch.tensor(keypoints_2d[:,:2] ) 
        target_tracking_info['dwpose_rlt']['kp3d'] = torch.zeros((44,4),dtype=torch.float32)  # [44,4]
        target_tracking_info['dwpose_rlt']['kp2d'] = torch.zeros((44,3),dtype=torch.float32)   # [44,3]
        

        source_ehm =  torchvision.transforms.functional.resize(image, (256, 256), antialias=True) # resize [3,518,518]
        source_image = torchvision.transforms.functional.resize(image, (518, 518), antialias=True) # resize [3,518,518]
        source_mask = torchvision.transforms.functional.resize(mask, (518, 518), antialias=True)
        target_image = torchvision.transforms.functional.resize(tagt_image, (512, 512), antialias=True) # resize [3,518,518]
        target_mask = torchvision.transforms.functional.resize(tagt_mask, (512, 512), antialias=True)
        target_ehm = torchvision.transforms.functional.resize(tagt_image, (256, 256), antialias=True) # resize [3,518,518]


        view_matrix,  full_proj_matrix  =  get_full_proj_matrix(target_tracking_info['w2c_cam'],  1/24)
        target_tracking_info['render_cam_params'] = {  # target info 多一个这个 和  mask.  目标渲染视角 
            "world_view_transform":view_matrix,  "full_proj_transform":full_proj_matrix,
            'tanfovx': 1/24,     'tanfovy': 1/24,
            'image_height':512,  'image_width':512,
            'camera_center':target_tracking_info['c2w_cam'][:3,3]
        }



        target_tracking_info['image'] = source_image               # [3,518,518] 518 resolution for ehm
        target_tracking_info['mask'] = source_mask                 # [1,518,518]
        target_tracking_info['source_ehm'] = source_ehm # [3,256,256]  256 resolution for ehm
        target_tracking_info['target_ehm'] = target_ehm  # 512 resolution for render supervisition

        target_tracking_info['target_image'] = target_image        # [ 3,512,512]  
        target_tracking_info['target_mask'] = target_mask          # [ 1,512,512]  
        target_tracking_info['image_name'] = sample['video_id']

        target_tracking_info['source_info'] = source_tracking_info





        return target_tracking_info
    





    def divided_dataset(sample):
        frame_id_list = sample.get( 'frame_id_list', None)
        if frame_id_list is not None:
        # dataset = dataset.map(decode_images)        # 图像转 PIL  可以晚一点解码出来
            source_tracking_info = example_formatter_all(sample)
        else:
            source_tracking_info = example_formatter_hmr(sample)

        return source_tracking_info

    dataset = dataset.map(divided_dataset)
    return dataset


# 过滤掉一些脏数据
# 比如：身体旋转过大，
# def apply_kp_filter(dataset:wds.WebDataset, cnt_thresh:int=4, conf_thresh:float=0.0):
#     '''
#     Counting the number of keypoints with confidence higher than the threshold.
#     If the number is less than the threshold, we regard it has insufficient valid 2D keypoints.
#     '''
#     if cnt_thresh > 0:
#         def insuff_kp_filter(item):
#             kp_conf = item['data']['kp2d'][:, 2]
#             return (kp_conf > conf_thresh).sum() > cnt_thresh
#         dataset = dataset.select(insuff_kp_filter)
#     return dataset


def load_box(tracking_info):
    crop_info={'body_crop':tracking_info['body_crop'],'head_crop':tracking_info['head_crop'],
                'left_hand_crop':tracking_info['left_hand_crop'],'right_hand_crop':tracking_info['right_hand_crop']}
    crop_info=data_to_tensor(crop_info)
    scale= 512/1024
    image_size=512
    head_crop_size=512
    hand_crop_size=512
    
    head_box_o=torch.tensor([[0.0,0.0,1.0],[head_crop_size,0.0,1.0],[0.0,head_crop_size,1.0],[head_crop_size,head_crop_size,1.0]])#x,y
    hand_box_o=torch.tensor([[0.0,0.0,1.0],[hand_crop_size,0.0,1.0],[0.0,hand_crop_size,1.0],[hand_crop_size,hand_crop_size,1.0]])#x,y
    
    body_crop=crop_info['body_crop']
    head_crop=crop_info['head_crop']
    left_hand_crop=crop_info['left_hand_crop']
    right_hand_crop=crop_info['right_hand_crop']
    
    head_box=body_crop['M_o2c-hd']@head_crop['M_c2o']@head_box_o[:,:,None]
    left_hand_box=body_crop['M_o2c-hd']@left_hand_crop['M_c2o']@hand_box_o[:,:,None]
    right_hand_box=body_crop['M_o2c-hd']@right_hand_crop['M_c2o']@hand_box_o[:,:,None]
    head_box*=scale
    left_hand_box*=scale
    right_hand_box*=scale
    head_box = head_box.clamp(0, image_size - 1)
    left_hand_box = left_hand_box.clamp(0, image_size - 1)
    right_hand_box = right_hand_box.clamp(0, image_size - 1)

    
    head_left,head_right=int(head_box.min(dim=0)[0][0]),int(head_box.max(dim=0)[0][0])
    head_top,head_bottom=int(head_box.min(dim=0)[0][1]),int(head_box.max(dim=0)[0][1])
    
    left_hand_left,left_hand_right=int(left_hand_box.min(dim=0)[0][0]),int(left_hand_box.max(dim=0)[0][0])
    left_hand_top,left_hand_bottom=int(left_hand_box.min(dim=0)[0][1]),int(left_hand_box.max(dim=0)[0][1])
    right_hand_left,right_hand_right=int(right_hand_box.min(dim=0)[0][0]),int(right_hand_box.max(dim=0)[0][0])
    right_hand_top,right_hand_bottom=int(right_hand_box.min(dim=0)[0][1]),int(right_hand_box.max(dim=0)[0][1])
    # left right top bottom
    head_box=torch.tensor([head_left,head_right,head_top,head_bottom],dtype=torch.long)
    left_hand_box=torch.tensor([left_hand_left,left_hand_right,left_hand_top,left_hand_bottom],dtype=torch.long)
    right_hand_box=torch.tensor([right_hand_left,right_hand_right,right_hand_top,right_hand_bottom],dtype=torch.long)
    if head_box[0]==head_box[1] or head_box[2]==head_box[3]:
        head_box=torch.tensor([0,image_size-1,0,image_size-1],dtype=torch.long)
    
    return head_box,left_hand_box,right_hand_box

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

    dataset = dataset.decode(pt_decoder)
    dataset = dataset.map(lambda x: x["pt"]["data"])  # 提取 dict
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
    # 每个 epoch 从数据集中读取最多 50,000 个样本, 1000: 指 shuffle 缓冲区的大小（buffer size）
    # initial=1000: 表示刚开始时先填满这个 buffer，再开始从中随机输出样本  
    # train_dataset 会在输出 第 50,000 个样本后自动触发 StopIteration；
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


    if split =='test' or split =='valid':
        return train_dataset
    # 每个 epoch 从数据集中读取最多 50,000 个样本, 1000: 指 shuffle 缓冲区的大小（buffer size）
    # initial=1000: 表示刚开始时先填满这个 buffer，再开始从中随机输出样本  
    # train_dataset 会在输出 第 50,000 个样本后自动触发 StopIteration；
    if Train:
        train_dataset = train_dataset.with_epoch(50_000).shuffle(1000, initial=1000)
    else:
        if shuffle:
            train_dataset = train_dataset.with_epoch(1000).shuffle(1000, initial=1000)
        else:
            train_dataset = train_dataset.with_epoch(1000)

    return train_dataset
