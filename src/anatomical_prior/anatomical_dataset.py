"""
Anatomical Refinement Dataset — extends MVHumanNet++ with GCN mesh priors.

This module wraps MVHumanNetData from mvhumannet.py.  We do NOT redefine the
camera, image, or depth loading logic — we inherit it and add two things:

  1. mesh_anchors [512, 3]  — posed SMPL-X vertices at the 512 canonical indices.
  2. normal_map   [3, H, W] — GT normal map for the Normal Consistency Loss.

ADJACENCY MATRIX — WHY IT IS FIXED
------------------------------------
SMPL-X is a template mesh. Its triangle topology (which vertex connects to which)
NEVER changes across poses, subjects, or frames. Only the (x,y,z) coordinates
vary with pose. Therefore:

  - Run FPS once on the T-pose (rest-pose) SMPL-X template → get 512 canonical indices.
  - Save those indices as fps_indices_512.npy (done once, reused everywhere).
  - Build the 512×512 adjacency matrix from those indices and the SMPL-X faces.
  - Save as smpl_adj_512.npy    (done once, reused everywhere).

  For every actual frame, load the SMPL-X parameters, run the body model forward,
  and index the resulting posed vertices with fps_indices_512.npy to get the
  512 current-frame anchors. The topology is always the same; only the positions change.

HOW TO GENERATE THE REQUIRED FILES (run once offline)
------------------------------------------------------
See scripts/preprocess_smplx_anchors.py for the full preprocessing pipeline:
  1. Load SMPL-X rest-pose → run FPS → save fps_indices_512.npy
  2. Build adjacency from SMPL-X faces + fps_indices → degree-normalise → save smpl_adj_512.npy
  3. For each sequence/frame: run SMPL-X forward with stored smplx_params →
     index posed vertices with fps_indices → save {frame}_anchors.npz

NORMAL MAP FORMAT
-----------------
  - Standard OpenGL/Blender tangent-space normal map PNG.
  - uint8, shape (H, W, 3), range [0, 255]:  pixel = (normal * 0.5 + 0.5) * 255
  - Stored at: {seq}/normals/{cam_id}/{frame_name}.png
  - The training loss remaps [0,1] → [-1,1] before cosine similarity.

ADDITIONAL FILE STRUCTURE (on top of existing MVHumanNet++)
------------------------------------------------------------
{seq_main}/
├── images/             (existing — unchanged)
├── depths/             (existing — unchanged)
├── cameras/            (existing — unchanged)
├── smplx_params/       (existing — unchanged)
└── normals/            ← NEW: one PNG per (cam_id, frame)
    └── cam_00/
    │   └── 0025.png
    └── cam_01/
        └── 0025.png

Dataset root:
└── smpl_adj_512.npy    ← NEW: [512,512] normalised adjacency (generated once)
└── fps_indices_512.npy ← NEW: [512] canonical vertex indices  (generated once)
"""

import logging
import os
import random

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from data.data import crop_resize_if_necessary
from data.mvhumannet.mvhumannet import (
    MVHumanNetData,
    worker_init_fn,
    smplx_collate_fn,
    NUM_CAMERAS,
    ALL_CAMERA_IDS,
    NUM_CONTEXT_VIEWS,
    NUM_TARGET_VIEWS,
)
from src.mast3r_src.dust3r.dust3r.datasets.utils.transforms import ImgNorm
from src.mast3r_src.dust3r.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_normal_map(path, target_hw):
    """
    Deprecated: Using crop_resize_if_necessary directly now.
    """
    pass


def _build_posed_anchors(smplx_params, fps_indices, smplx_body_model):
    """
    Run the SMPL-X body model forward with the stored per-frame parameters
    to produce posed vertex positions, then slice the 512 canonical anchors.

    Args:
        smplx_params:      dict from MVHumanNetData.get_smplx()  (numpy arrays)
        fps_indices:       [512] int array of canonical vertex indices
        smplx_body_model:  smplx.SMPLX model (CPU, no_grad)

    Returns:
        anchors: float32 numpy [512, 3] in world/camera space
    """
    def _t(x):
        return torch.from_numpy(x).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        out = smplx_body_model(
            betas=_t(smplx_params['betas']),
            global_orient=_t(smplx_params['global_orient']),
            body_pose=_t(smplx_params['body_pose']),
            left_hand_pose=_t(smplx_params.get('left_hand_pose',
                              np.zeros(6, np.float32))),
            right_hand_pose=_t(smplx_params.get('right_hand_pose',
                               np.zeros(6, np.float32))),
            jaw_pose=_t(smplx_params.get('jaw_pose', np.zeros(3, np.float32))),
            leye_pose=_t(smplx_params.get('leye_pose', np.zeros(3, np.float32))),
            reye_pose=_t(smplx_params.get('reye_pose', np.zeros(3, np.float32))),
            expression=_t(smplx_params.get('expression', np.zeros(10, np.float32))),
            transl=_t(smplx_params.get('transl', np.zeros(3, np.float32))),
            return_verts=True,
        )
    vertices = out.vertices[0].cpu().numpy()  # [10475, 3]
    return vertices[fps_indices].astype(np.float32)  # [512, 3]


# ---------------------------------------------------------------------------
# Extended Data Indexer
# ---------------------------------------------------------------------------

class MVHumanNetAnatomicalData(MVHumanNetData):
    """
    Extends MVHumanNetData with normal map and SMPL-X anchor loading.

    Top-level roots (matching DATASET_FORMAT.md):
        main_root   = main/{seq}/...
        depth_root  = depth/{seq}/...
        normal_root = normal/{seq}/{cam}/{frame}.png
        anchor_root = anchors/{seq}/{frame}.npz        key='anchors'
    """

    def __init__(
        self,
        main_root: str,
        depth_root: str,
        normal_root: str,
        anchor_root: str,
        smplx_model_path: str,
        fps_indices_path: str,
        sequences=None,
    ):
        super().__init__(main_root, depth_root, sequences)
        self.normal_root = normal_root
        self.anchor_root = anchor_root

        # Fixed 512 canonical vertex indices (computed on T-pose, reused for all frames)
        self.fps_indices = np.load(fps_indices_path).astype(int)  # [512]
        assert len(self.fps_indices) == 512, "Expected exactly 512 FPS indices."

        # No need to load the SMPL-X model.
        # We precomputed all 512-vertex anchors as NPZ files inside anchors/
        self.smplx_model = None

    def get_normal(self, seq, cam_id, frame_name, resolution):
        """
        Load GT normal map and apply identical crop+resize as RGB images.
        """
        path = os.path.join(self.normal_root, seq, 'normal', cam_id, f'{frame_name}.jpg')
        if not os.path.exists(path):
            path = os.path.join(self.normal_root, seq, 'normal', cam_id, f'{frame_name}.png')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Strict Check: Normal map not found at {path}")
                
        img = Image.open(path).convert('RGB')
        W, H = img.size
        dummy_depth = np.zeros((H, W), dtype=np.float32)
        
        K = self.cameras[seq][cam_id]['K'].copy()
        from data.data import crop_resize_if_necessary
        img, _, _ = crop_resize_if_necessary(img, dummy_depth, K, resolution)
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)

    def get_anchors(self, seq, frame_name):
        """
        Load precomputed posed SMPL-X anchors from anchors/{seq}/{frame_name}.npz
        NPZ key expected: 'anchors'  shape [512, 3] float32
        """
        anchor_path = os.path.join(self.anchor_root, seq, f'{frame_name}.npz')
        if not os.path.exists(anchor_path):
            raise FileNotFoundError(f"Strict Check: Anchor file missing at {anchor_path}. Model requires valid human prior anchors dataset.")
            
        return np.load(anchor_path)['anchors'].astype(np.float32)  # key='anchors'

    def get_mask(self, seq, cam_id, frame_name, resolution):
        """
        Load human mask from masks/{cam_id}/{frame_name}.png and apply identical crop+resize.
        Returns a boolean array of shape (H, W).
        """
        path = os.path.join(self.main_root, seq, 'masks', cam_id, f'{frame_name}.png')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Strict Check: Mask not found at {path}")
            
        img = Image.open(path).convert('L')
        W, H = img.size
        dummy_depth = np.zeros((H, W), dtype=np.float32)
        
        K = self.cameras[seq][cam_id]['K'].copy()
        from data.data import crop_resize_if_necessary
        img, _, _ = crop_resize_if_necessary(img, dummy_depth, K, resolution)
        return np.array(img) > 127


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class MVHumanNetAnatomicalDataset(torch.utils.data.Dataset):
    """
    Training dataset for Anatomical Refinement (GCN + Cross-Attention).

    Extends MVHumanNetSplattingDataset with two extra fields:
      - context view: mesh_anchors [512, 3]
      - target  view: normal_map   [3, H, W]

    Everything else (image loading, depth, camera, pts3d) is unchanged.

    DDP note: smpl_adj is returned per-sample (same tensor every time since it
    is fixed). Lightning's default collate will stack it to [B, 512, 512];
    training code takes [0] to get [512, 512] and broadcasts.
    """

    def __init__(
        self,
        data: MVHumanNetAnatomicalData,
        adj: torch.Tensor,
        resolution,
        num_context_views: int = NUM_CONTEXT_VIEWS,
        num_target_views:  int = NUM_TARGET_VIEWS,
        num_epochs_per_epoch: int = 1,
    ):
        super().__init__()
        self.data              = data
        self.adj               = adj                   # [512, 512], fixed
        self.resolution        = resolution
        self.num_context_views = num_context_views
        self.num_target_views  = num_target_views

        self.transform     = ImgNorm
        self.org_transform = torchvision.transforms.ToTensor()

        base_index = [
            (seq, frame_name)
            for seq in data.sequences
            for frame_name in data.frame_names[seq]
        ]
        self.index = base_index * num_epochs_per_epoch
        logger.info(
            f"[MVHumanNetAnatomicalDataset] "
            f"{len(base_index)} base items × {num_epochs_per_epoch} = {len(self.index)} total."
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq, frame_name = self.index[idx]

        cam_order    = random.sample(ALL_CAMERA_IDS, len(ALL_CAMERA_IDS))
        context_cams = cam_order[:self.num_context_views]
        target_cams  = cam_order[self.num_context_views:
                                  self.num_context_views + self.num_target_views]

        # Load mesh anchors ONCE per frame (same across all cameras for the same frame)
        anchors = self.data.get_anchors(seq, frame_name)  # [512, 3]

        views = {'context': [], 'target': [], 'scene': seq, 'smpl_adj': self.adj}

        # ── Context views ─────────────────────────────────────────────────────
        for cam_id in context_cams:
            view = self.data.get_view(seq, cam_id, frame_name, self.resolution)

            view['img']          = self.transform(view['original_img'])
            view['original_img'] = self.org_transform(view['original_img'])

            pts3d, valid_mask    = depthmap_to_absolute_camera_coordinates(**view)
            view['pts3d']        = pts3d
            
            mask_human = self.data.get_mask(seq, cam_id, frame_name, self.resolution)
            view['mask_human'] = mask_human
            view['valid_mask']   = valid_mask & np.isfinite(pts3d).all(axis=-1) & mask_human

            # Anchor vertices: same for all cameras of this frame
            view['mesh_anchors'] = torch.from_numpy(anchors)  # [512, 3]

            views['context'].append(view)

        # ── Target views ──────────────────────────────────────────────────────
        for cam_id in target_cams:
            view = self.data.get_view(seq, cam_id, frame_name, self.resolution)
            view['original_img'] = self.org_transform(view['original_img'])

            # Mask and Normal maps applying identical deterministic crop to align with original img
            view['mask_human'] = self.data.get_mask(seq, cam_id, frame_name, self.resolution)
            view['normal_map'] = self.data.get_normal(seq, cam_id, frame_name, self.resolution)

            views['target'].append(view)

        views['smplx'] = self.data.get_smplx(seq, frame_name)
        return views


# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------

class MVHumanNetAnatomicalTestDataset(torch.utils.data.Dataset):
    """
    Deterministic evaluation dataset. Mirrors MVHumanNetSplattingTestDataset.
    samples: list of (seq, cam_ctx_1, cam_ctx_2, cam_target, frame_name) tuples.
    """

    def __init__(self, data: MVHumanNetAnatomicalData, adj, samples, resolution):
        super().__init__()
        self.data          = data
        self.adj           = adj
        self.samples       = samples
        self.resolution    = resolution
        self.transform     = ImgNorm
        self.org_transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, cam_ctx_1, cam_ctx_2, cam_target, frame_name = self.samples[idx]
        anchors = self.data.get_anchors(seq, frame_name)  # [512, 3]

        def _ctx(cam_id):
            view = self.data.get_view(seq, cam_id, frame_name, self.resolution)
            view['img']          = self.transform(view['original_img'])
            view['original_img'] = self.org_transform(view['original_img'])
            pts3d, valid_mask    = depthmap_to_absolute_camera_coordinates(**view)
            view['pts3d']        = pts3d
            mask_human           = self.data.get_mask(seq, cam_id, frame_name, self.resolution)
            view['mask_human']   = mask_human
            view['valid_mask']   = valid_mask & np.isfinite(pts3d).all(axis=-1) & mask_human
            view['mesh_anchors'] = torch.from_numpy(anchors)
            return view

        ctx1 = _ctx(cam_ctx_1)
        ctx2 = _ctx(cam_ctx_2)

        target = self.data.get_view(seq, cam_target, frame_name, self.resolution)
        target['original_img'] = self.org_transform(target['original_img'])
        target['mask_human'] = self.data.get_mask(seq, cam_target, frame_name, self.resolution)
        target['normal_map'] = self.data.get_normal(seq, cam_target, frame_name, self.resolution)

        return {
            'context':  [ctx1, ctx2],
            'target':   [target],
            'scene':    seq,
            'smpl_adj': self.adj,
            'smplx':    self.data.get_smplx(seq, frame_name),
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def get_anatomical_dataset(
    main_root: str,
    depth_root: str,
    normal_root: str,
    anchor_root: str,
    adj_path: str,
    fps_indices_path: str,
    smplx_model_path: str,
    resolution,
    sequences=None,
    num_epochs_per_epoch: int = 1,
):
    """
    Build an AnatomicalRefinement training dataset.
    All root paths match the structure in DATASET_FORMAT.md.
    """
    data = MVHumanNetAnatomicalData(
        main_root, depth_root, normal_root, anchor_root,
        smplx_model_path, fps_indices_path, sequences
    )
    adj = torch.from_numpy(np.load(adj_path).astype(np.float32))
    return MVHumanNetAnatomicalDataset(
        data=data, adj=adj, resolution=resolution,
        num_epochs_per_epoch=num_epochs_per_epoch,
    )


def get_anatomical_test_dataset(
    main_root: str,
    depth_root: str,
    normal_root: str,
    anchor_root: str,
    adj_path: str,
    fps_indices_path: str,
    smplx_model_path: str,
    resolution,
    samples: list,
    sequences=None,
):
    """
    Build a deterministic evaluation dataset.
    samples: list of (seq, cam_ctx_1, cam_ctx_2, cam_target, frame_name) tuples.
    """
    data = MVHumanNetAnatomicalData(
        main_root, depth_root, normal_root, anchor_root,
        smplx_model_path, fps_indices_path, sequences
    )
    adj = torch.from_numpy(np.load(adj_path).astype(np.float32))
    return MVHumanNetAnatomicalTestDataset(data, adj, samples, resolution)
