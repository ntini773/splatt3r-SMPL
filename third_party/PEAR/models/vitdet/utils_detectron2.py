

from tqdm import tqdm

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate as instantiate_detectron2
from detectron2.data import MetadataCatalog
from typing import List, Tuple, Union
from  omegaconf import OmegaConf
import torch
import numpy as np

import cv2


def to_tensor(x, device, temporary:bool=False):
    '''
    Simply unify the type transformation to torch.Tensor. 
    If device is None, don't change the device if device is not CPU. 
    '''
    if isinstance(x, torch.Tensor):
        device = x.device if device is None else device
        if temporary:
            recover_type_back = lambda x_: x_.to(x.device)  # recover the device
            return x.to(device), recover_type_back
        else:
            return x.to(device)

    device = 'cpu' if device is None else device
    if isinstance(x, np.ndarray):
        if temporary:
            recover_type_back = lambda x_: x_.detach().cpu().numpy()
            return torch.from_numpy(x).to(device), recover_type_back
        else:
            return torch.from_numpy(x).to(device)
    if isinstance(x, List):
        if temporary:
            recover_type_back = lambda x_: x_.tolist()
            return torch.from_numpy(np.array(x)).to(device), recover_type_back
        else:
            return torch.from_numpy(np.array(x)).to(device)
    raise ValueError(f"Unsupported type: {type(x)}")


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




class DefaultPredictor_Lazy:
    '''
    Create a simple end-to-end predictor with the given config that runs on single device for a
    several input images.
    Compared to using the model directly, this class does the following additions:

    Modified from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/utils/utils_detectron2.py#L9-L93

    1. Load checkpoint from the weights specified in config (cfg.MODEL.WEIGHTS).
    2. Always take BGR image as the input and apply format conversion internally.
    3. Apply resizing defined by the parameter `max_img_size`.
    4. Take input images and produce outputs, and filter out only the `instances` data.
    5. Use an auto-tuned batch size to process the images in a batch.
        - Start with the given batch size, if failed, reduce the batch size by half.
        - If the batch size is reduced to 1 and still failed, skip the image.
        - The implementation is abstracted to `lib.platform.sliding_batches`.
    '''

    def __init__(self, cfg, batch_size=20, max_img_size=512, device='cuda:0'):
        self.batch_size = batch_size
        self.max_img_size = max_img_size
        self.device = device
        self.model = instantiate_detectron2(cfg.model)

        test_dataset = OmegaConf.select(cfg, 'dataloader.test.dataset.names', default=None)
        if isinstance(test_dataset, (List, Tuple)):
            test_dataset = test_dataset[0]

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(OmegaConf.select(cfg, 'train.init_checkpoint', default=''))

        mapper = instantiate_detectron2(cfg.dataloader.test.mapper)
        self.aug = mapper.augmentations
        self.input_format = mapper.image_format

        self.model.eval().to(self.device)
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)

        assert self.input_format in ['RGB'], f'Invalid input format: {self.input_format}'
        # assert self.input_format in ['RGB', 'BGR'], f'Invalid input format: {self.input_format}'

    def __call__(self, imgs):
        '''
        ### Args
        - `imgs`: List[np.ndarray], a list of image of shape (Hi, Wi, RGB). 
            - Shapes of each image may be different.

        ### Returns
        - `predictions`: dict,
            - the output of the model for one image only.
            - See :doc:`/tutorials/models` for details about the format.
        '''
        with torch.no_grad():
            inputs = []
            downsample_ratios = []
            for img in imgs:
                img_size = max(img.shape[:2])
                if img_size > self.max_img_size:  # exceed the max size, make it smaller
                    downsample_ratio = self.max_img_size / img_size
                    img = flex_resize_img(img, ratio=downsample_ratio)
                    downsample_ratios.append(downsample_ratio)
                else:
                    downsample_ratios.append(1.0)
                h, w, _ = img.shape
                img = self.aug(T.AugInput(img)).apply_image(img)
                img = to_tensor(img.astype('float32').transpose(2, 0, 1), 'cpu')
                inputs.append({'image': img, 'height': h, 'width': w})

            preds = []
            N_imgs = len(inputs)
            prog_bar = tqdm(total=N_imgs, desc='Batch Detection')
            sid, last_fail_id = 0, 0
            cur_bs = self.batch_size
            while sid < N_imgs:
                eid = min(sid + cur_bs, N_imgs)
                try:
                    preds_round = self.model(inputs[sid:eid]) # 把一个batch 图像输入
                except Exception as e:
                    if cur_bs > 1:
                        cur_bs = (cur_bs - 1) // 2 + 1  # reduce the batch size by half
                        assert cur_bs > 0, 'Invalid batch size.'
                    else:
                        preds.append(None)  # placeholder for the failed image
                        sid += 1
                    last_fail_id = sid
                    continue
                # Save the results.
                preds.extend([{
                        'pred_classes' : pred['instances'].pred_classes.cpu(),
                        'scores'       : pred['instances'].scores.cpu(),
                        'pred_boxes'   : pred['instances'].pred_boxes.tensor.cpu(),
                    } for pred in preds_round])

                prog_bar.update(eid - sid)
                sid = eid
                # # Adjust the batch size.
                # if last_fail_id < sid - cur_bs * 2:
                #     cur_bs = min(cur_bs * 2, self.batch_size)  # gradually recover the batch size
            prog_bar.close()

            return preds, downsample_ratios
