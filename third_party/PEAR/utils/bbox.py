import torch
from models.modules.ehm import EHM_v2 
from models.pipeline.ehm_pipeline import Ehm_Pipeline
import os
import torch
from utils.pipeline_utils import to_tensor
from utils.graphics_utils import GS_Camera
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from pytorch3d.renderer import PointLights
import cv2

import os
import torch
import argparse
import lightning
import numpy as np
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import (
    ConfigDict, rtqdm, device_parser, add_extra_cfgs
)
import glob
import tqdm

from typing import Union, Optional, Tuple,List



def flex_resize_video(
    frames : np.ndarray,
    tgt_wh : Union[Tuple[int, int], None] = None,
    ratio  : Union[float, None] = None,
    kp_mod : int = 1,
):
    '''
    Resize the frames to the target width and height. Set one of width and height to -1 to keep the aspect ratio.
    Only one of `tgt_wh` and `ratio` can be set, if both are set, `tgt_wh` will be used.

    ### Args
    - frames: np.ndarray, (L, H, W, 3)
    - tgt_wh: Tuple[int, int], default=None
        - The target width and height, set one of them to -1 to keep the aspect ratio.
    - ratio: float, default=None
        - The ratio to resize the frames. It will be used if `tgt_wh` is not set.
    - kp_mod: int, default 1
        - Keep the width and height as multiples of `kp_mod`.
        - For example, if `kp_mod=16`, the width and height will be rounded to the nearest multiple of 16.

    ### Returns
    - np.ndarray, (L, H', W', 3)
        - The resized frames.
    '''
    assert tgt_wh is not None or ratio is not None, 'At least one of tgt_wh and ratio must be set.'
    if tgt_wh is not None:
        assert len(tgt_wh) == 2, 'tgt_wh must be a tuple of 2 elements.'
        assert tgt_wh[0] > 0 or tgt_wh[1] > 0, 'At least one of width and height must be positive.'
    if ratio is not None:
        assert ratio > 0, 'ratio must be positive.'
    assert len(frames.shape) == 4, 'frames must have 3 or 4 dimensions.'

    def align_size(val:float):
        ''' It will round the value to the nearest multiple of `kp_mod`. '''
        return int(round(val / kp_mod) * kp_mod)

    # Calculate the target width and height.
    orig_h, orig_w = frames.shape[1], frames.shape[2]
    tgt_wh = (int(orig_w * ratio), int(orig_h * ratio)) if tgt_wh is None else tgt_wh  # Get wh from ratio if not given. # type: ignore
    tgt_w, tgt_h = tgt_wh
    tgt_w = align_size(orig_w * tgt_h / orig_h) if tgt_w == -1 else align_size(tgt_w)
    tgt_h = align_size(orig_h * tgt_w / orig_w) if tgt_h == -1 else align_size(tgt_h)
    # Resize the frames.
    resized_frames = np.stack([cv2.resize(frame, (tgt_w, tgt_h)) for frame in frames])

    return resized_frames



def flex_resize_img(
    img    : np.ndarray,
    tgt_wh : Union[Tuple[int, int], None] = None,
    ratio  : Union[float, None] = None,
    kp_mod : int = 1,
):
    '''
    Resize the image to the target width and height. Set one of width and height to -1 to keep the aspect ratio.
    Only one of `tgt_wh` and `ratio` can be set, if both are set, `tgt_wh` will be used.

    ### Args
    - img: np.ndarray, (H, W, 3)
    - tgt_wh: Tuple[int, int], default=None
        - The target width and height, set one of them to -1 to keep the aspect ratio.
    - ratio: float, default=None
        - The ratio to resize the frames. It will be used if `tgt_wh` is not set.
    - kp_mod: int, default 1
        - Keep the width and height as multiples of `kp_mod`.
        - For example, if `kp_mod=16`, the width and height will be rounded to the nearest multiple of 16.

    ### Returns
    - np.ndarray, (H', W', 3)
        - The resized iamges.
    '''
    assert len(img.shape) == 3, 'img must have 3 dimensions.'
    return flex_resize_video(img[None], tgt_wh, ratio, kp_mod)[0]


def lurb_to_cwh(
    lurb : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the left-upper-right-bottom format to the center-width-height format.

    ### Args
    - lurb: Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The left-upper-right-bottom format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The center-width-height format bounding box.
    '''
    lurb, recover_type_back = to_tensor(lurb, device=None, temporary=True)
    assert lurb.shape[-1] == 4, f"Invalid shape: {lurb.shape}, should be (..., 4)"

    c = (lurb[..., :2] + lurb[..., 2:]) / 2  # (..., 2)
    wh = lurb[..., 2:] - lurb[..., :2]  # (..., 2)

    cwh = torch.cat([c, wh], dim=-1)  # (..., 4)
    return recover_type_back(cwh)


def cwh_to_lurb(
    cwh : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the center-width-height format to the left-upper-right-bottom format.

    ### Args
    - cwh: Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The center-width-height format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The left-upper-right-bottom format bounding box.
    '''
    cwh, recover_type_back = to_tensor(cwh, device=None, temporary=True)
    assert cwh.shape[-1] == 4, f"Invalid shape: {cwh.shape}, should be (..., 4)"

    l = cwh[..., :2] - cwh[..., 2:] / 2  # (..., 2)
    r = cwh[..., :2] + cwh[..., 2:] / 2  # (..., 2)

    lurb = torch.cat([l, r], dim=-1)  # (..., 4)
    return recover_type_back(lurb)


def cwh_to_cs(
    cwh    : Union[list, np.ndarray, torch.Tensor],
    reduce : Optional[str] = None,
):
    '''
    Convert the center-width-height format to the center-scale format.
    *Only works when width and height are the same.*

    ### Args
    - cwh: Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The center-width-height format bounding box.
    - reduce: Optional[str], default None, valid values: None, 'max'
        - Determine how to reduce the width and height to a single scale.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 3)
        - The center-scale format bounding box.
    '''
    cwh, recover_type_back = to_tensor(cwh, device=None, temporary=True)
    assert cwh.shape[-1] == 4, f"Invalid shape: {cwh.shape}, should be (..., 4)"

    if reduce is None:
        if (cwh[..., 2] != cwh[..., 3]).any():
            print(f"Width and height are supposed to be the same, but they're not. The larger one will be used.")

    c = cwh[..., :2]  # (..., 2)
    s = cwh[..., 2:].max(dim=-1)[0]  # (...,)  最大的那个

    cs = torch.cat([c, s[..., None]], dim=-1)  # (..., 3)
    return recover_type_back(cs)


def cs_to_cwh(
    cs : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the center-scale format to the center-width-height format.

    ### Args
    - cs: Union[list, np.ndarray, torch.Tensor], (..., 3)
        - The center-scale format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The center-width-height format bounding box.
    '''
    cs, recover_type_back = to_tensor(cs, device=None, temporary=True)
    assert cs.shape[-1] == 3, f"Invalid shape: {cs.shape}, should be (..., 3)"

    c = cs[..., :2]  # (..., 2)
    s = cs[..., 2]  # (...,)

    cwh = torch.cat([c, s[..., None], s[..., None]], dim=-1)  # (..., 4)
    return recover_type_back(cwh)


def lurb_to_cs(
    lurb : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the left-upper-right-bottom format to the center-scale format.
    *Only works when width and height are the same.*

    ### Args
    - lurb: Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The left-upper-right-bottom format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 3)
        - The center-scale format bounding box.
    '''
    return cwh_to_cs(lurb_to_cwh(lurb), reduce='max') # 先转化为 center  w h 还是一个4参数


def cs_to_lurb(
    cs : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the center-scale format to the left-upper-right-bottom format.

    ### Args
    - cs: Union[list, np.ndarray, torch.Tensor], (..., 3)
        - The center-scale format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor], (..., 4)
        - The left-upper-right-bottom format bounding box.
    '''
    return cwh_to_lurb(cs_to_cwh(cs))


def lurb_to_luwh(
    lurb : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the left-upper-right-bottom format to the left-upper-width-height format.

    ### Args
    - lurb: Union[list, np.ndarray, torch.Tensor]
        - The left-upper-right-bottom format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor]
        - The left-upper-width-height format bounding box.
    '''
    lurb, recover_type_back = to_tensor(lurb, device=None, temporary=True)
    assert lurb.shape[-1] == 4, f"Invalid shape: {lurb.shape}, should be (..., 4)"

    lu = lurb[..., :2]  # (..., 2)
    wh = lurb[..., 2:] - lurb[..., :2]  # (..., 2)

    luwh = torch.cat([lu, wh], dim=-1)  # (..., 4)
    return recover_type_back(luwh)


def luwh_to_lurb(
    luwh : Union[list, np.ndarray, torch.Tensor],
):
    '''
    Convert the left-upper-width-height format to the left-upper-right-bottom format.

    ### Args
    - luwh: Union[list, np.ndarray, torch.Tensor]
        - The left-upper-width-height format bounding box.

    ### Returns
    - Union[list, np.ndarray, torch.Tensor]
        - The left-upper-right-bottom format bounding box.
    '''
    luwh, recover_type_back = to_tensor(luwh, device=None, temporary=True)
    assert luwh.shape[-1] == 4, f"Invalid shape: {luwh.shape}, should be (..., 4)"

    l = luwh[..., :2]  # (..., 2)
    r = luwh[..., :2] + luwh[..., 2:]  # (..., 2)

    lurb = torch.cat([l, r], dim=-1)  # (..., 4)
    return recover_type_back(lurb)


def crop_with_lurb(data, lurb, padding=0):
    """
    Crop the img-like data according to the lurb bounding box.
    
    ### Args
    - data: Union[np.ndarray, torch.Tensor], shape (H, W, C)
        - Data like image.
    - lurb: Union[list, np.ndarray, torch.Tensor], shape (4,)
        - Bounding box with [left, upper, right, bottom] coordinates.
    - padding: int, default 0
        - Padding value for out-of-bound areas.
        
    ### Returns
    - Union[np.ndarray, torch.Tensor], shape (H', W', C)
        - Cropped image with padding if necessary.
    """
    data, recover_type_back = to_tensor(data, device=None, temporary=True)

    # Ensure lurb is in numpy array format for indexing
    lurb = np.array(lurb).astype(np.int64)
    l_, u_, r_, b_ = lurb

    # Determine the shape of the data.
    H_raw, W_raw, C_raw = data.size()

    # Compute the cropped patch size.
    H_patch = b_ - u_
    W_patch = r_ - l_

    # Create an output buffer of the crop size, initialized to padding
    if isinstance(data, np.ndarray):
        output = np.full((H_patch, W_patch, C_raw), padding, dtype=data.dtype)
    else:
        output = torch.full((H_patch, W_patch, C_raw), padding, dtype=data.dtype)

    # Calculate the valid region in the original data
    valid_l_ = max(0, l_)
    valid_u_ = max(0, u_)
    valid_r_ = min(W_raw, r_)
    valid_b_ = min(H_raw, b_)

    # Calculate the corresponding valid region in the output
    target_l_ = valid_l_ - l_
    target_u_ = valid_u_ - u_
    target_r_ = target_l_ + (valid_r_ - valid_l_)
    target_b_ = target_u_ + (valid_b_ - valid_u_)

    # Copy the valid region into the output buffer
    output[target_u_:target_b_, target_l_:target_r_, :] = data[valid_u_:valid_b_, valid_l_:valid_r_, :]

    return recover_type_back(output)


def fit_bbox_to_aspect_ratio(
    bbox      : np.ndarray,
    tgt_ratio : Optional[Tuple[int, int]] = None,
    bbox_type : str = 'lurb'
):
    '''
    Fit a random bounding box to a target aspect ratio through enlarging the bounding box with least change.
    
    ### Args
    - bbox: np.ndarray, shape is determined by `bbox_type`, e.g. for 'lurb', shape is (4,)
        - The bounding box to be modified. The format is determined by `bbox_type`.
    - tgt_ratio: Optional[Tuple[int, int]], default None
        - The target aspect ratio to be matched.
    - bbox_type: str, default 'lurb', valid values: 'lurb', 'cwh'.
    
    ### Returns
    - np.ndarray, shape is determined by `bbox_type`, e.g. for 'lurb', shape is (4,)
        - The modified bounding box.
    '''
    bbox = bbox.copy()
    if bbox_type == 'lurb':
        bbx_cwh = lurb_to_cwh(bbox)
        bbx_wh = bbx_cwh[2:]
    elif bbox_type == 'cwh':
        bbx_wh = bbox[2:]
    else:
        raise ValueError(f"Unsupported bbox type: {bbox_type}")

    new_bbx_wh = expand_wh_to_aspect_ratio(bbx_wh, tgt_ratio)

    if bbox_type == 'lurb':
        bbx_cwh[2:] = new_bbx_wh
        new_bbox = cwh_to_lurb(bbx_cwh)
    elif bbox_type == 'cwh':
        new_bbox = np.concatenate([bbox[:2], new_bbx_wh])
    else:
        raise ValueError(f"Unsupported bbox type: {bbox_type}")

    return new_bbox

def to_numpy(x, temporary:bool=False):
    if isinstance(x, torch.Tensor):
        if temporary:
            recover_type_back = lambda x_: torch.from_numpy(x_).type_as(x).to(x.device)
            return x.detach().cpu().numpy(), recover_type_back
        else:
            return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if temporary:
            recover_type_back = lambda x_: x_
            return x.copy(), recover_type_back
        else:
            return x
    if isinstance(x, List):
        if temporary:
            recover_type_back = lambda x_: x_.tolist()
            return np.array(x), recover_type_back
        else:
            return np.array(x)
    raise ValueError(f"Unsupported type: {type(x)}")


def expand_wh_to_aspect_ratio(bbx_wh:np.ndarray, tgt_aspect_ratio:Optional[Tuple[int, int]]=None):
    '''
    Increase the size of the bounding box to match the target shape.
    Modified from https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L14-L33
    '''
    if tgt_aspect_ratio is None:
        return bbx_wh

    try:
        bbx_w , bbx_h = bbx_wh
    except (ValueError, TypeError):
        print(f"Invalid bbox_wh content: {bbx_wh}")
        return bbx_wh

    tgt_w, tgt_h = tgt_aspect_ratio
    if bbx_h / bbx_w < tgt_h / tgt_w:
        new_h = max(bbx_w * tgt_h / tgt_w, bbx_h)
        new_w = bbx_w
    else:
        new_h = bbx_h
        new_w = max(bbx_h * tgt_w / tgt_h, bbx_w)
    assert new_h >= bbx_h and new_w >= bbx_w

    return to_numpy([new_w, new_h])

def fit_bbox_to_aspect_ratio(
    bbox      : np.ndarray,
    tgt_ratio : Optional[Tuple[int, int]] = None,
    bbox_type : str = 'lurb'
):
    '''
    Fit a random bounding box to a target aspect ratio through enlarging the bounding box with least change.
    
    ### Args
    - bbox: np.ndarray, shape is determined by `bbox_type`, e.g. for 'lurb', shape is (4,)
        - The bounding box to be modified. The format is determined by `bbox_type`.
    - tgt_ratio: Optional[Tuple[int, int]], default None
        - The target aspect ratio to be matched.
    - bbox_type: str, default 'lurb', valid values: 'lurb', 'cwh'.
    
    ### Returns
    - np.ndarray, shape is determined by `bbox_type`, e.g. for 'lurb', shape is (4,)
        - The modified bounding box.
    '''
    bbox = bbox.copy()
    if bbox_type == 'lurb':
        bbx_cwh = lurb_to_cwh(bbox)
        bbx_wh = bbx_cwh[2:]
    elif bbox_type == 'cwh':
        bbx_wh = bbox[2:]
    else:
        raise ValueError(f"Unsupported bbox type: {bbox_type}")

    new_bbx_wh = expand_wh_to_aspect_ratio(bbx_wh, tgt_ratio)

    if bbox_type == 'lurb':
        bbx_cwh[2:] = new_bbx_wh
        new_bbox = cwh_to_lurb(bbx_cwh)
    elif bbox_type == 'cwh':
        new_bbox = np.concatenate([bbox[:2], new_bbx_wh])
    else:
        raise ValueError(f"Unsupported bbox type: {bbox_type}")

    return new_bbox

def _img_det2patches(imgs, det_instances, downsample_ratio:float, max_instances:int=5):
    '''
    1. Filter out the trusted human detections.
    2. Enlarge the bounding boxes to aspect ratio (ViT backbone only use 192*256 pixels, make sure these 
       pixels can capture main contents) and then to squares (to adapt the data module).
    3. Crop the image with the bounding boxes and resize them to 256x256.
    4. Normalize the cropped images.
    '''
    if det_instances is None:  # no human detected
        return to_numpy([]), to_numpy([])
    CLASS_HUMAN_ID, DET_THRESHOLD_SCORE = 0, 0.5

    # Filter out the trusted human detections.
    is_human_mask = det_instances['pred_classes'] == CLASS_HUMAN_ID
    reliable_mask = det_instances['scores'] > DET_THRESHOLD_SCORE
    active_mask = is_human_mask & reliable_mask

    # Filter out the top-k human instances.
    if active_mask.sum().item() > max_instances:  # 最多 5 个人体
        humans_scores = det_instances['scores'] * is_human_mask.float()
        _, top_idx = humans_scores.topk(max_instances)
        valid_mask = torch.zeros_like(active_mask).bool()
        valid_mask[top_idx] = True
    else:
        valid_mask = active_mask

    # Process the bounding boxes and crop the images.
    lurb_all = det_instances['pred_boxes'][valid_mask].numpy() / downsample_ratio  # (N, 4)  左上角-右下角格式（LURB）边界框。
    lurb_all = [fit_bbox_to_aspect_ratio(bbox=lurb, tgt_ratio=(192, 256)) for lurb in lurb_all]  # regularize the bbox size 把 bbox 调整为特定宽高比的矩形
    # 将 LURB 格式 bbox 转换成 center-scale 格式（中心坐标 + 尺度），常见于人体估计任务。
    cs_all   = [lurb_to_cs(lurb) for lurb in lurb_all]  # convert rectangle left-up-right-bottom bbx to square center-scale bbx 
    #又将 center-scale 还原为标准的 [x1, y1, x2, y2] 格式。也就是正方形的格式
    lurb_all = [cs_to_lurb(cs) for cs in cs_all]  # convert square center-scale bbx to rectangle left-up-right-bottom bbx 
    cropped_imgs = [crop_with_lurb(imgs, lurb) for lurb in lurb_all]  # crop_with_lurb()：使用 LURB 形式的 bbox 去从原图 imgs 中裁剪出目标区域。
    # flex_resize_img()：将每个裁剪图像 resize 到统一的大小 (256, 256)。
    patches = to_numpy([flex_resize_img(cropped_img, (256, 256)) for cropped_img in cropped_imgs])  # (N, 256, 256, 3) 
    return patches, cs_all

def imgs_det2patches(imgs, dets, downsample_ratios, max_instances_per_img):
    ''' Given the raw images and the detection results, return the image patches of human instances. '''
    assert len(imgs) == len(dets), f'L_img = {len(imgs)}, L_det = {len(dets)}'
    patches, n_patch_per_img, bbx_cs = [], [], []
    for i in tqdm(range(len(imgs))):
        patches_i, bbx_cs_i = _img_det2patches(imgs[i], dets[i], downsample_ratios[i], max_instances_per_img)
        n_patch_per_img.append(len(patches_i))
        if len(patches_i) > 0:
            patches.append(patches_i.astype(np.float32))
            bbx_cs.append(bbx_cs_i)
        else:
            print(f'No human detection results on image No.{i}.')
    det_meta = {
            'n_patch_per_img' : n_patch_per_img,
            'bbx_cs'          : bbx_cs,
        }
    return patches, det_meta  # 返回人体部分的 图像，并且resize成了 256 大小



def load_img_meta(
    img_path : Union[str, Path],
):
    ''' Read the image meta from the given path without opening image. '''
    assert Path(img_path).exists(), f'Image not found: {img_path}'
    H, W = imageio.v3.improps(img_path).shape[:2]
    meta = {'w': W, 'h': H}
    return meta


def load_img(
    img_path : Union[str, Path],
    mode     : str = 'RGB',
):
    ''' Read the image from the given path. '''
    assert Path(img_path).exists(), f'Image not found: {img_path}'

    img = imageio.v3.imread(img_path, plugin='pillow', mode=mode)

    meta = {
        'w': img.shape[1],
        'h': img.shape[0],
    }
    return img, meta


from pathlib import Path

def load_inputs(input_path, input_type, MAX_IMG_W=1920, MAX_IMG_H=1080):
    # 1. Inference inputs type.
    inputs_path = Path(input_path)
    if input_type != 'auto': inputs_type = input_type
    else: inputs_type = 'video' if Path(input_path).is_file() else 'imgs'
    print(f'🚚 Loading inputs from: {inputs_path}, regarded as <{inputs_type}>.')

    # 2. Load inputs.
    inputs_meta = {'type': inputs_type}
    if inputs_type == 'video':
        inputs_meta['seq_name'] = inputs_path.stem
        frames, _ = load_video(inputs_path)
        if frames.shape[1] > MAX_IMG_H:
            frames = flex_resize_video(frames, (MAX_IMG_H, -1), kp_mod=4)
        if frames.shape[2] > MAX_IMG_W:
            frames = flex_resize_video(frames, (-1, MAX_IMG_W), kp_mod=4)
        raw_imgs = [frame for frame in frames]
    elif inputs_type == 'imgs':
        img_fns = list(inputs_path.glob('*.*'))
        img_fns = [fn for fn in img_fns if fn.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        inputs_meta['seq_name'] = f'{inputs_path.stem}-img_cnt={len(img_fns)}'
        raw_imgs = []
        for fn in img_fns:
            img, _ = load_img(fn)
            if img.shape[0] > MAX_IMG_H:
                img = flex_resize_img(img, (MAX_IMG_H, -1), kp_mod=4)
            if img.shape[1] > MAX_IMG_W:
                img = flex_resize_img(img, (-1, MAX_IMG_W), kp_mod=4)
            raw_imgs.append(img)
        inputs_meta['img_fns'] = img_fns
    else:
        raise ValueError(f'Unsupported inputs type: {inputs_type}.')
    print(f'📦 Totally {len(raw_imgs)} images are loaded.')

    return raw_imgs, inputs_meta


import imageio

def load_video(
    video_path : Union[str, Path],
):
    ''' Read the video from the given path. '''
    if isinstance(video_path, str):
        video_path = Path(video_path)

    assert video_path.exists(), f'Video not found: {video_path}'

    if video_path.is_dir():
        print(f'Found {video_path} is a directory. It will be regarded as a image folder.')
        imgs_path = sorted(glob(str(video_path / '*')))
        frames = []
        for img_path in tqdm(imgs_path):
            frames.append(imageio.imread(img_path))
        fps = 30 # default fps
    else:
        print(f'Found {video_path} is a file. It will be regarded as a video file.')
        reader = imageio.get_reader(video_path, format='FFMPEG')
        frames = []
        for frame in tqdm(reader, total=reader.count_frames()):
            frames.append(frame)
        fps = reader.get_meta_data()['fps']
    frames = np.stack(frames, axis=0) # (L, H, W, 3)
    meta = {
        'fps': fps,
        'w'  : frames.shape[2],
        'h'  : frames.shape[1],
        'L'  : frames.shape[0],
    }

    return frames, meta
