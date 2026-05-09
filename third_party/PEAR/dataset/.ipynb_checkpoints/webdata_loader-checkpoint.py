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



class MixedWebDataset(wds.WebDataset):
    def __init__(self) -> None:
        super(wds.WebDataset, self).__init__()


def pt_decoder(key, value):
    if key.endswith(".pt"):
        return {"data": torch.load(io.BytesIO(value), map_location='cpu')}
    else:
        return {key: value}



to_tensor = transforms.ToTensor()  # 转换为 [C, H, W] 的 float32 tensor，范围 [0,1]

def decode_images(sample):
    """将 bytes 图像转为 torch.Tensor，范围 [0,1]"""
    img = Image.open(io.BytesIO(sample["body_image"])).convert("RGB")
    mask = Image.open(io.BytesIO(sample["body_mask"])).convert("L")  # 如果 mask 是灰度图

    sample["body_image"] = to_tensor(img)   # [3, H, W], float32, [0,1]
    sample["body_mask"] = to_tensor(mask)   # [1, H, W], float32, [0,1]

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


def apply_example_formatter(dataset, cfg):
    def fet_tracking_info_from_raw(raw_item):
        
        tracking_info =  deepcopy(raw_item['params'])
        tracking_info['smplx_coeffs'].update({"shape":raw_item['id_params']['smplx_shape'][0],  # 加入一些共有的信息
                                                "joints_offset":raw_item['id_params']['joints_offset'][0],
                                                "head_scale":raw_item['id_params']['head_scale'][0],
                                                "hand_scale":raw_item['id_params']['hand_scale'][0],
                                                })
        hmr_pose = tracking_info['smplx_coeffs'].get("hmr_pose")
        if hmr_pose is not None:
            tracking_info['smplx_coeffs']["body_pose"][[0,1,3,4,6,7,9,10]] = hmr_pose[[0,1,3,4,6,7,9,10]]
        else:
            tracking_info['smplx_coeffs']["hmr_pose"] = tracking_info['smplx_coeffs']["body_pose"]
        tracking_info['flame_coeffs'].update({"shape_params":raw_item['id_params']['flame_shape'][0]})

        tracking_info=data_to_tensor(tracking_info)  # 为啥有两个？
        tracking_info=squeeze_params(tracking_info)

        # tracking_info=data_to_tensor(tracking_info)
        # tracking_info=squeeze_params(tracking_info)
        
        # Convert PyTorch 3D coordinate system to the COLMAP coordinate system. 
        # (Since it is identical to the image coordinate, the same camera parameters 
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

    def example_formatter(raw_item):
        # TODO 6.27  这里做个归一化
        source_tracking_info = fet_tracking_info_from_raw(raw_item)  # 获取跟踪信息
        original_image,  source_mask = raw_item['body_image'], raw_item['body_mask']

        source_image = original_image  # * source_mask  + (1-source_mask) * self.bg_color  再将人体抠出来  似乎没必要把人体抠出来
        ehm_source_image = torchvision.transforms.functional.resize(source_image, (256, 256), antialias=True)
        source_image = torchvision.transforms.functional.resize(source_image, (518, 518), antialias=True) # resize [3,518,518]
        source_mask = torchvision.transforms.functional.resize(source_mask, (518, 518), antialias=True)

        target_mask = torchvision.transforms.functional.resize(source_mask, (512, 512), antialias=True)
        target_image = torchvision.transforms.functional.resize(original_image, (512, 512), antialias=True) # resize [3,518,518]


        view_matrix,  full_proj_matrix  =  get_full_proj_matrix(source_tracking_info['w2c_cam'],  1/24)

        source_tracking_info['render_cam_params']={  # target info 多一个这个 和  mask.  目标渲染视角 
            "world_view_transform":view_matrix,"full_proj_transform":full_proj_matrix,
            'tanfovx': 1/24,'tanfovy': 1/24,
            'image_height':1024,  'image_width':1024,
            'camera_center':source_tracking_info['c2w_cam'][:3,3]
        }

        source_tracking_info['image'] = source_image         # 518 resolution for ehm
        source_tracking_info['mask'] = source_mask
        source_tracking_info['ehm_image'] = ehm_source_image  # 256 resolution for ehm
        source_tracking_info['target_image'] = target_image  # 512 resolution for render supervisition
        source_tracking_info['target_mask'] = target_mask

        return source_tracking_info

    dataset = dataset.map(example_formatter)
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

def load_tars_as_wds(cfg, urls,  resampled,  train = True):

    urls = expand_urls(urls)  # to list of URL strings

    dataset : wds.WebDataset = wds.WebDataset(
            urls, # 所有数据位置
            nodesplitter = wds.split_by_node,  # 分布式环境下的数据分片器
            shardshuffle = True,  # 是否对 .tar 文件顺序打乱
            resampled    = True,  # resampled, # 每个 epoch 从数据集中采样 resampled 个样本，流式
            cache_dir    = None,  # 	是否启用缓存，默认关闭
        )

    dataset = dataset.decode(pt_decoder)
    dataset = dataset.map(lambda x: x["pt"]["data"])  # 提取 dict
    dataset = dataset.map(decode_images)        # 图像转 PIL
    dataset = apply_example_formatter(dataset, cfg)

    return dataset





def build_web_tracked_data(cfg_dataset, split):
    names, datasets, weights = [], [], []
    if split == 'train':
        dataset_type = cfg_dataset.datasets
        Train = True
    else:
        dataset_type = cfg_dataset.val_datasets
        Train = False  # For debug

    for ds_cfg in dataset_type:  # Iterate over all datasets in the training configuration.
        dataset = load_tars_as_wds(
                cfg_dataset,
                ds_cfg.item.urls,
                ds_cfg.item.epoch_size,
                Train
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
    # 每个 epoch 从数据集中读取最多 50,000 个样本, 1000: 指 shuffle 缓冲区的大小（buffer size）
    # initial=1000: 表示刚开始时先填满这个 buffer，再开始从中随机输出样本  
    # train_dataset 会在输出 第 50,000 个样本后自动触发 StopIteration；
    if Train:
        train_dataset = train_dataset.with_epoch(50_000).shuffle(1000, initial=1000)
    else:
        train_dataset = train_dataset.with_epoch(1000).shuffle(1000, initial=1000)

    return train_dataset
