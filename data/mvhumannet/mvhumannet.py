import logging
import os
import random

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from data.data import crop_resize_if_necessary
from src.mast3r_src.dust3r.dust3r.datasets.utils.transforms import ImgNorm
from src.mast3r_src.dust3r.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

logger = logging.getLogger(__name__)

NUM_CAMERAS = 16
ALL_CAMERA_IDS = [f"cam_{i:02d}" for i in range(NUM_CAMERAS)]
NUM_CONTEXT_VIEWS = 2
NUM_TARGET_VIEWS = 1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_exr_depth(path: str) -> np.ndarray:
    """
    Load a depth map from an EXR file.
    Returns a float32 H×W array in metres.
    Tries imageio.v3 first; falls back to OpenCV (requires EXR support built-in).
    """
    depth = None
    try:
        import imageio.v3 as iio
        depth = np.array(iio.imread(path))
    except Exception:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise IOError(f"Failed to load EXR depth from: {path}")

    # EXR may be multi-channel (e.g. RGB float) — take first channel
    if depth.ndim == 3:
        depth = depth[..., 0]

    return depth.astype(np.float32)


def w2c_to_c2w(extrinsic: np.ndarray) -> np.ndarray:
    """
    Convert a world-to-camera matrix [R|t] of shape (3, 4) in OpenCV convention
    (Y-down, Z-forward) to a camera-to-world matrix of shape (4, 4).

    OpenCV convention is already what dust3r / MASt3R expects internally,
    so no axis flip is needed here (unlike ScanNetPP which needed a P @ c2w @ P.T
    conversion from nerfstudio's OpenGL Y-up convention).

    Math:
        w2c: p_cam = R @ p_world + t
        c2w: [ R^T | -R^T @ t ]
             [  0  |     1    ]
    """
    R = extrinsic[:3, :3].astype(np.float32)
    t = extrinsic[:3,  3].astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T
    c2w[:3,  3] = -R.T @ t
    return c2w


# ---------------------------------------------------------------------------
# Data Indexer
# ---------------------------------------------------------------------------

class MVHumanNetData:
    """
    Indexes all sequences, cameras, and SMPL-X parameters at init time.
    Images and depth maps are loaded lazily on demand via get_view().

    Separating indexing (this class) from the PyTorch Dataset allows the
    small camera/SMPL-X metadata to be loaded once per worker process and
    reused across all __getitem__ calls (important with persistent_workers=True).

    Args:
        main_root:   Path to the 'main/' split — contains images/, cameras/,
                     annots/, masks/, smplx_params/ per sequence.
        depth_root:  Path to the 'depth/' split — contains depths/ per sequence.
                     Kept separate so both can live on different storage mounts
                     (e.g. SSD for images, NAS for depth) and to support future
                     datasets that may have a different depth root.
        sequences:   Optional explicit list of sequence IDs (strings).
                     If None, all subdirectories of main_root are used.
    """

    def __init__(self, main_root: str, depth_root: str, sequences=None):
        self.main_root  = main_root
        self.depth_root = depth_root

        if sequences is None:
            self.sequences = sorted([
                d for d in os.listdir(main_root)
                if os.path.isdir(os.path.join(main_root, d))
            ])
        else:
            self.sequences = list(sequences)

        # Populated per sequence in _index_sequence()
        self.frame_names = {}   # seq → list[str]  e.g. ['0025', '0050', ..., '1500']
        self.cameras     = {}   # seq → {cam_id: {'K': (4,4), 'c2w': (4,4)}}

        for seq in self.sequences:
            self._index_sequence(seq)

        logger.info(
            f"[MVHumanNetData] Indexed {len(self.sequences)} sequences, "
            f"{NUM_CAMERAS} cameras each."
        )

    def _index_sequence(self, seq: str):
        seq_main = os.path.join(self.main_root, seq)

        # ── Frame names ───────────────────────────────────────────────────────
        # Derive from cam_00 image listing — names are identical across all cams.
        # Skip 'A-*' frames (augmented / anomalous frames with irregular naming).
        cam00_dir = os.path.join(seq_main, 'images', 'cam_00')
        self.frame_names[seq] = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(cam00_dir)
            if f.endswith('.jpg') and not f.startswith('A-')
        ])

        # ── Camera calibrations ───────────────────────────────────────────────
        self.cameras[seq] = {}
        for cam_id in ALL_CAMERA_IDS:
            cam_npz = np.load(
                os.path.join(seq_main, 'cameras', cam_id, 'camera.npz')
            )

            # Intrinsics: (3,3) float64 → pad to (4,4) float32
            K = np.eye(4, dtype=np.float32)
            K[:3, :3] = cam_npz['intrinsic'].astype(np.float32)

            # Extrinsics: w2c (3,4) → c2w (4,4), OpenCV convention preserved
            c2w = w2c_to_c2w(cam_npz['extrinsic'])

            self.cameras[seq][cam_id] = {'K': K, 'c2w': c2w}

        logger.debug(
            f"  seq={seq}: {len(self.frame_names[seq])} frames indexed."
        )

    # ------------------------------------------------------------------
    # Public API — called inside Dataset.__getitem__
    # ------------------------------------------------------------------

    def get_view(self, seq: str, cam_id: str, frame_name: str, resolution) -> dict:
        """
        Load one (image, depth, camera) triplet, apply crop/resize,
        and return the standard view dict expected by the Splatt3r pipeline.

        Path layout:
            image: main_root/{seq}/images/{cam_id}/{frame_name}.jpg
            depth: depth_root/{seq}/depths/{cam_id}/{frame_name}.exr
        """
        seq_main = os.path.join(self.main_root, seq)

        # RGB — opened as PIL for crop_resize_if_necessary compatibility
        img_path = os.path.join(seq_main, 'images', cam_id, f'{frame_name}.jpg')
        image = Image.open(img_path).convert('RGB')

        # Depth — float32 H×W in metres
        depth_path = os.path.join(
            self.depth_root, seq, 'depths', cam_id, f'{frame_name}.exr'
        )
        depthmap = load_exr_depth(depth_path)

        # Camera — copies so cached arrays are not mutated by crop_resize
        K   = self.cameras[seq][cam_id]['K'].copy()
        c2w = self.cameras[seq][cam_id]['c2w'].copy()

        # Crop to principal point, then Lanczos-downscale, then crop to resolution.
        # Adjusts K to match the new image dimensions.
        image, depthmap, K = crop_resize_if_necessary(image, depthmap, K, resolution)

        return {
            'original_img':      image,           # PIL Image — caller converts to tensor
            'depthmap':          depthmap,        # (H, W) float32, metres
            'camera_pose':       c2w,             # (4, 4) float32, c2w OpenCV
            'camera_intrinsics': K,               # (4, 4) float32
            'dataset':           'mvhumannet++',
            'label':             f'mvhumannet++/{seq}',
            'instance':          f'{cam_id}_{frame_name}',
            'is_metric_scale':   True,
            'sky_mask':          depthmap <= 0.0, # (H, W) bool — invalid/background pixels
        }

    def get_smplx(self, seq: str, frame_name: str) -> dict:
        """
        Load per-frame SMPL-X parameters from smplx_params/{frame_name}.npz.

        Uses per-frame files (not the aggregated smplx_params.npz) because:
          - They are the original fitter output and have more fields
            (scale, leye_pose, reye_pose, pose).
          - File name matches image/depth name directly — no index arithmetic.

        Returns a flat dict of float32 numpy arrays with the batch dim squeezed:
            global_orient  (3,)    root orientation in world space
            transl         (3,)    root XYZ translation in world space
            body_pose      (63,)   21 body joints × 3 axis-angle
            betas          (10,)   body shape (same person → constant across frames)
            jaw_pose       (3,)
            left_hand_pose (6,)    PCA-compressed (6 components → 45 joint angles internally)
            right_hand_pose(6,)
            expression     (10,)   FLAME face expression coefficients
            leye_pose      (3,)    left eyeball rotation
            reye_pose      (3,)    right eyeball rotation
            scale          (1,)    global scale from fitter
            pose           (66,)   global_orient(3) + body_pose(63) concatenated
        """
        smplx_path = os.path.join(
            self.main_root, seq, 'smplx_params', f'{frame_name}.npz'
        )
        raw = np.load(smplx_path, allow_pickle=True)
        # squeeze removes the batch dim (1,...) → (...,) for all arrays
        return {k: raw[k].squeeze(0).astype(np.float32) for k in raw.files}


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class MVHumanNetSplattingDataset(torch.utils.data.Dataset):
    """
    Training dataset for Splatt3r + SMPL-X on MVHumanNet++.

    Sampling strategy:
        Each item = one time frame from one sequence.
        Camera views are sampled by randomly shuffling all 16 cameras:
          first num_context_views  → context  (full: img + pts3d + pose)
          next  num_target_views   → target   (image + pose only)
        No coverage precomputation is needed because all cameras are
        synchronized — same-frame views always have scene overlap.

    DDP compatibility:
        The flat (seq, frame_name) index is what Lightning's DistributedSampler
        splits across GPUs. Each GPU gets a disjoint subset of (seq, frame)
        pairs. Camera shuffling inside __getitem__ is seeded per-worker via
        worker_init_fn (defined below) so workers don't pick identical cameras.

    Args:
        data:                 MVHumanNetData instance.
        resolution:           (H, W) target resolution.
        num_context_views:    How many cameras to use as context input.
        num_target_views:     How many cameras to render as target.
        num_epochs_per_epoch: Repeats the flat index N times to make each
                              Lightning epoch larger (same trick as ScanNetPP —
                              avoids overhead from very short epochs).
    """

    def __init__(
        self,
        data: MVHumanNetData,
        resolution,
        num_context_views:    int = NUM_CONTEXT_VIEWS,
        num_target_views:     int = NUM_TARGET_VIEWS,
        num_epochs_per_epoch: int = 1,
    ):
        super().__init__()
        self.data              = data
        self.resolution        = resolution
        self.num_context_views = num_context_views
        self.num_target_views  = num_target_views

        self.transform     = ImgNorm                            # ImageNet-normalised for encoder
        self.org_transform = torchvision.transforms.ToTensor() # [0, 1] float for loss

        # Flat index: each entry = (seq, frame_name).
        # Multiplied by num_epochs_per_epoch to control epoch granularity.
        base_index = [
            (seq, frame_name)
            for seq in data.sequences
            for frame_name in data.frame_names[seq]
        ]
        self.index = base_index * num_epochs_per_epoch

        logger.info(
            f"[MVHumanNetSplattingDataset] "
            f"{len(base_index)} base items × {num_epochs_per_epoch} = {len(self.index)} total."
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        seq, frame_name = self.index[idx]

        # ── Camera sampling ───────────────────────────────────────────────────
        cam_order    = random.sample(ALL_CAMERA_IDS, len(ALL_CAMERA_IDS))
        context_cams = cam_order[:self.num_context_views]
        target_cams  = cam_order[self.num_context_views:
                                  self.num_context_views + self.num_target_views]

        views = {'context': [], 'target': [], 'scene': seq}

        # ── Context views ─────────────────────────────────────────────────────
        # Encoder input: requires normalised img + 3D point cloud + validity mask.
        for cam_id in context_cams:
            view = self.data.get_view(seq, cam_id, frame_name, self.resolution)

            # Two image representations:
            #   img          → ImageNet-normalised tensor, fed to the encoder
            #   original_img → [0,1] tensor, used for SH residual & rendering loss
            view['img']          = self.transform(view['original_img'])
            view['original_img'] = self.org_transform(view['original_img'])

            # Unproject depth → 3D points in world space (used by MAST3R loss)
            pts3d, valid_mask    = depthmap_to_absolute_camera_coordinates(**view)
            view['pts3d']        = pts3d
            view['valid_mask']   = valid_mask & np.isfinite(pts3d).all(axis=-1)

            assert view['valid_mask'].any(), (
                f"All depth invalid: seq={seq}, cam={cam_id}, frame={frame_name}"
            )

            views['context'].append(view)

        # ── Target views ──────────────────────────────────────────────────────
        # Gaussian renderer renders these views; only original_img + camera needed.
        for cam_id in target_cams:
            view = self.data.get_view(seq, cam_id, frame_name, self.resolution)
            view['original_img'] = self.org_transform(view['original_img'])
            views['target'].append(view)

        # ── SMPL-X ground truth ───────────────────────────────────────────────
        # Passed through the batch; consumed only by the SMPL-X head's loss.
        # Gaussian pipeline (context/target) is completely unaffected.
        views['smplx'] = self.data.get_smplx(seq, frame_name)

        return views


# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------

class MVHumanNetSplattingTestDataset(torch.utils.data.Dataset):
    """
    Evaluation dataset with fixed (seq, context_cam_1, context_cam_2, target_cam)
    tuples — mirrors DUST3RSplattingTestDataset from ScanNetPP.

    samples: list of (seq, cam_ctx_1, cam_ctx_2, cam_target) strings.
    """

    def __init__(self, data: MVHumanNetData, samples: list, resolution):
        super().__init__()
        self.data          = data
        self.samples       = samples
        self.resolution    = resolution
        self.transform     = ImgNorm
        self.org_transform = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_context_view(self, seq, cam_id, frame_name):
        view = self.data.get_view(seq, cam_id, frame_name, self.resolution)
        view['img']          = self.transform(view['original_img'])
        view['original_img'] = self.org_transform(view['original_img'])
        pts3d, valid_mask    = depthmap_to_absolute_camera_coordinates(**view)
        view['pts3d']        = pts3d
        view['valid_mask']   = valid_mask & np.isfinite(pts3d).all(axis=-1)
        assert view['valid_mask'].any(), (
            f"All depth invalid: seq={seq}, cam={cam_id}, frame={frame_name}"
        )
        return view

    def __getitem__(self, idx: int) -> dict:
        seq, cam_ctx_1, cam_ctx_2, cam_target, frame_name = self.samples[idx]

        ctx1   = self._load_context_view(seq, cam_ctx_1, frame_name)
        ctx2   = self._load_context_view(seq, cam_ctx_2, frame_name)

        target = self.data.get_view(seq, cam_target, frame_name, self.resolution)
        target['original_img'] = self.org_transform(target['original_img'])

        return {
            'context': [ctx1, ctx2],
            'target':  [target],
            'scene':   seq,
            'smplx':   self.data.get_smplx(seq, frame_name),
        }


# ---------------------------------------------------------------------------
# Worker seeding — required for correct DDP + multi-worker behaviour
# ---------------------------------------------------------------------------

def worker_init_fn(worker_id: int):
    """
    Seed Python random and NumPy differently per DataLoader worker.

    Without this, all workers share the same base seed after fork, causing
    every worker to pick the same camera shuffle for its assigned items.
    Pass this as `worker_init_fn=worker_init_fn` in the DataLoader.
    """
    seed = (torch.initial_seed() % (2 ** 32)) + worker_id
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def get_mvhumannet_dataset(
    main_root:            str,
    depth_root:           str,
    resolution,
    sequences=None,
    num_epochs_per_epoch: int = 1,
    num_context_views:    int = NUM_CONTEXT_VIEWS,
    num_target_views:     int = NUM_TARGET_VIEWS,
) -> MVHumanNetSplattingDataset:
    """
    Build a training dataset.

    Args:
        main_root:            Path to the 'main/' folder.
        depth_root:           Path to the 'depth/' folder (may differ from main_root).
        resolution:           (H, W) target crop/resize resolution.
        sequences:            Explicit sequence list or None for auto-discovery.
        num_epochs_per_epoch: Index repeat multiplier (see class docstring).
    """
    data = MVHumanNetData(main_root, depth_root, sequences)
    return MVHumanNetSplattingDataset(
        data=data,
        resolution=resolution,
        num_context_views=num_context_views,
        num_target_views=num_target_views,
        num_epochs_per_epoch=num_epochs_per_epoch,
    )


def get_mvhumannet_test_dataset(
    main_root:  str,
    depth_root: str,
    resolution,
    samples:    list,
    sequences=None,
) -> MVHumanNetSplattingTestDataset:
    """
    Build a test/evaluation dataset from a fixed sample list.

    Args:
        samples: list of (seq, cam_ctx_1, cam_ctx_2, cam_target, frame_name) tuples.
    """
    data = MVHumanNetData(main_root, depth_root, sequences)
    return MVHumanNetSplattingTestDataset(data, samples, resolution)


def smplx_collate_fn(batch_list):
    """
    PyTorch collate function that handles:
    1. Stacking 'context' and 'target' views (lists of dicts).
    2. Stacking 'smplx' parameters (nested dict of tensors).
    3. Collecting 'scene' names (list of strings).
    """
    assert len(batch_list) > 0
    
    out = {
        'context': [],
        'target':  [],
        'scene':   [b['scene'] for b in batch_list],
    }

    # 1. Collate Views (Context/Target)
    for key in ['context', 'target']:
        num_views = len(batch_list[0][key])
        for v_idx in range(num_views):
            view_batch = {}
            # All items in batch have the same keys for a given view index
            sample_view = batch_list[0][key][v_idx]
            for k, v in sample_view.items():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    # Convert to tensor if numpy, then stack
                    tensors = []
                    for b in batch_list:
                        val = b[key][v_idx][k]
                        if isinstance(val, np.ndarray):
                            val = torch.from_numpy(val)
                        tensors.append(val)
                    view_batch[k] = torch.stack(tensors, dim=0)
                else:
                    # Collect as list (strings/labels/etc)
                    view_batch[k] = [b[key][v_idx][k] for b in batch_list]
            out[key].append(view_batch)

    # 2. Collate SMPL-X
    if 'smplx' in batch_list[0]:
        out['smplx'] = {}
        for k in batch_list[0]['smplx'].keys():
            tensors = []
            for b in batch_list:
                val = b['smplx'][k]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                tensors.append(val)
            out['smplx'][k] = torch.stack(tensors, dim=0)

    return out
