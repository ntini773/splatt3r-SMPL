"""
N-View Mesh-Aware Inference Pipeline with GCN Conditioning

Pipeline:
1. Load 4-5 view batches with mesh_anchors + image_paths
2. Optional retrieval-based image pairing
3. Run sparse_global_alignment with integrated GCN mesh conditioning
4. Render target views using gsplat gaussian splatting
5. Export rendered images + cameras + metrics
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import workspace

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src', 'mast3r_src'))
sys.path.insert(0, os.path.join(current_dir, 'src', 'mast3r_src', 'dust3r'))

# Imports
try:
    from train_anatomical_refinement import MAST3RAnatomicalRefinement
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment, forward_mast3r
    from mast3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from mast3r.utils.misc import hash_md5
    from torch.utils.data import DataLoader
    from PIL import Image
    import cv2
    from plyfile import PlyData, PlyElement
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

# Optional retrieval
try:
    from mast3r.retrieval.processor import Retriever
    HAS_RETRIEVAL = True
except ImportError as e:
    HAS_RETRIEVAL = False
    print(f"⚠ Retrieval module not available due to {e}")

# Optional gsplat
try:
    from gsplat.rendering import rasterization
    HAS_GSPLAT = True
except ImportError:
    HAS_GSPLAT = False
    print("⚠ gsplat not available - will use basic rendering")

# Import N-view dataset
try:
    from src.anatomical_prior.anatomical_dataset import (
        get_anatomical_nview_dataset,
        smplx_collate_fn,
    )
except ImportError as e:
    print(f"Error importing dataset: {e}")
    sys.exit(1)


def load_gaussian_attributes(cache_path: str, img1_instance: str, img2_instance: str, is_img1: bool = True):
    """Load cached gaussian attributes."""
    idx1 = hash_md5(img1_instance)
    idx2 = hash_md5(img2_instance)
    
    if is_img1:
        path_gauss = os.path.join(cache_path, 'forward', idx1, f'{idx2}_gauss.pth')
    else:
        path_gauss = os.path.join(cache_path, 'forward', idx2, f'{idx1}_gauss.pth')
    
    if os.path.isfile(path_gauss):
        gauss_attrs = torch.load(path_gauss)
        return gauss_attrs[0] if is_img1 else gauss_attrs[1]
    return None


def _resolve_image_size(img_obj: Dict, default_hw: Tuple[int, int] = (512, 512)) -> Tuple[int, int]:
    """Robustly resolve image size as (H, W) from loader metadata."""
    shape = img_obj.get('true_shape', default_hw)

    # Normalize tensor/ndarray -> python list
    if isinstance(shape, torch.Tensor):
        shape = shape.detach().cpu().tolist()
    elif isinstance(shape, np.ndarray):
        shape = shape.tolist()

    # Unwrap nested containers like [[H, W]]
    while isinstance(shape, (list, tuple)) and len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Tensor, np.ndarray)):
        shape = shape[0]
        if isinstance(shape, torch.Tensor):
            shape = shape.detach().cpu().tolist()
        elif isinstance(shape, np.ndarray):
            shape = shape.tolist()

    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        h, w = int(shape[0]), int(shape[1])
        return h, w

    return default_hw


def _find_gaussian_attrs_for_view(cache_dir: str, img_instance: str, all_instances: List[str]):
    """Find gaussian attrs for one view by probing hashed cache paths.

    SparseGA cache naming uses instance IDs hashed with md5 and stores pair files
    under `forward/<hash(i)>/<hash(j)>_gauss.pth`.
    """
    if img_instance is None:
        return None
    instance_str = str(img_instance)

    for other in all_instances:
        other = str(other)
        if other == instance_str:
            continue

        # Case 1: current view is first image in pair file
        attrs = load_gaussian_attributes(cache_dir, instance_str, other, is_img1=True)
        if attrs is not None:
            return attrs

        # Case 2: current view is second image in pair file
        attrs = load_gaussian_attributes(cache_dir, other, instance_str, is_img1=False)
        if attrs is not None:
            return attrs

    return None


def _save_simple_splat_ply(path: str, xyz: np.ndarray, opacity: np.ndarray, scale: np.ndarray, rot: np.ndarray, sh: np.ndarray):
    """Saves Gaussian Splats to a .ply file compatible with 3DGS viewers."""
    print(f"  Saving {len(xyz)} splats to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    if sh.ndim == 3:
        sh = sh[:, 0, :]
    
    # Ensure sh has 3 channels (RGB)
    if sh.shape[-1] == 1:
        sh = np.repeat(sh, 3, axis=-1)
    elif sh.shape[-1] != 3:
        if sh.shape[-1] > 3:
            sh = sh[:, :3]
        else:
            padding = np.zeros((sh.shape[0], 3 - sh.shape[-1]))
            sh = np.concatenate([sh, padding], axis=-1)
    
    elements = np.empty(len(xyz), dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0
    elements['f_dc_0'] = sh[:, 0]
    elements['f_dc_1'] = sh[:, 1]
    elements['f_dc_2'] = sh[:, 2]
    elements['opacity'] = 1 / (1 + np.exp(-opacity[:, 0])) 
    elements['scale_0'] = np.log(np.clip(scale[:, 0], 1e-6, None))
    elements['scale_1'] = np.log(np.clip(scale[:, 1], 1e-6, None))
    elements['scale_2'] = np.log(np.clip(scale[:, 2], 1e-6, None))
    elements['rot_0'] = rot[:, 0] 
    elements['rot_1'] = rot[:, 1] 
    elements['rot_2'] = rot[:, 2] 
    elements['rot_3'] = rot[:, 3] 

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def prepare_mesh_anchors_dict(batch: Dict, file_paths: List[str]) -> Dict:
    """
    Build mesh_anchors dict keyed by file path string (matching img['instance'] set
    by convert_dust3r_pairs_naming in sparse_global_alignment).
    """
    mesh_anchors = {}
    shared_adj = None

    if 'smpl_adj' in batch:
        shared_adj = batch['smpl_adj'][0]  # [512, 512]
        print(f"  ✓ Loaded adjacency matrix: shape {shared_adj.shape}")
    else:
        print("  ⚠ Warning: 'smpl_adj' not in batch, GCN conditioning will be disabled")

    if 'context' in batch:
        for idx, view in enumerate(batch['context']):
            mesh = view['mesh_anchors'][0] if isinstance(view['mesh_anchors'], list) else view['mesh_anchors']
            # Key MUST be the file path — forward_mast3r looks up mesh_anchors.get(img1['instance'])
            # and img['instance'] is set to the file path by convert_dust3r_pairs_naming.
            if idx < len(file_paths):
                mesh_key = file_paths[idx]
                mesh_anchors[mesh_key] = {
                    'anchors': mesh.cpu().numpy() if isinstance(mesh, torch.Tensor) else mesh,
                    'adj': shared_adj.cpu().numpy() if isinstance(shared_adj, torch.Tensor) else shared_adj,
                }

    return mesh_anchors


def load_model_for_inference(
    model_ckpt: str,
    device: str,
    config: Optional[dict] = None,
    avoid_base_load: bool = True,
) -> torch.nn.Module:
    """Load the anatomical refinement checkpoint for inference.

    By default (`avoid_base_load=True`) this will instantiate the model
    via `MAST3RAnatomicalRefinement(config)` (when `config` is provided)
    and load the checkpoint state dict into it. That prevents any
    implicit behavior in `load_from_checkpoint` that might try to
    re-load a base Splatt3R checkpoint referenced by the model's
    configuration.

    If `config` is not provided we fall back to `load_from_checkpoint`.
    """
    if config is not None and avoid_base_load:
        model = MAST3RAnatomicalRefinement(config)
        ckpt = torch.load(model_ckpt, map_location='cpu')
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        missing_unexpected = model.load_state_dict(state_dict, strict=False)
        # `load_state_dict` may return a NamedTuple or tuple depending on torch version
        try:
            missing, unexpected = missing_unexpected
        except Exception:
            missing, unexpected = None, None
        if missing:
            print(f'Warning: missing keys while loading checkpoint: {len(missing)}')
        if unexpected:
            print(f'Warning: unexpected keys while loading checkpoint: {len(unexpected)}')
        print(f"✓ Loaded checkpoint with custom model instantiation: {model_ckpt}")
    else:
        # Fallback: behavior identical to previous code path
        model = MAST3RAnatomicalRefinement.load_from_checkpoint(
            model_ckpt,
            map_location='cpu',
            strict=False,
        )

    model.to(device).eval()
    try:
        model.lpips_criterion = lpips.LPIPS('vgg', spatial=True).to(device).eval()
    except Exception:
        # lpips optional; ignore if unavailable
        pass

    print("✓ Loaded MAST3RAnatomicalRefinement checkpoint")
    return model


def render_gaussian_splats(
    means: np.ndarray,           # [N, 3]
    scales: np.ndarray,          # [N, 3]
    rotations: np.ndarray,       # [N, 4]
    opacities: np.ndarray,       # [N, 1]
    sh: np.ndarray,              # [N, 3] or [N, C]
    intrinsic: np.ndarray,       # [3, 3]
    extrinsic: np.ndarray,       # [3, 4] or [4, 4]
    image_size: Tuple[int, int], # (H, W)
    device: str = 'cuda',
) -> np.ndarray:
    """Render gaussian splats using gsplat (if available) or basic projection."""
    
    if not HAS_GSPLAT:
        # Fallback: project means to image and use simple rendering
        H, W = image_size
        if extrinsic.shape[0] == 4:
            R, t = extrinsic[:3, :3], extrinsic[:3, 3]
        else:
            R, t = extrinsic[:3, :3], extrinsic[:3, 3]
        
        # Transform points to camera frame
        points_cam = (means @ R.T) + t
        
        # Project to image
        points_2d = points_cam @ intrinsic.T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        # Create output image with SH colors
        image = np.ones((H, W, 3), dtype=np.float32) * 0.5
        
        # Simple splatting: render each gaussian as a small blob
        for i in range(min(len(means), 100000)):  # Limit for speed
            x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
            if 0 <= x < W and 0 <= y < H:
                color = sh[i] if sh.ndim == 2 else sh[i, :3]
                color = np.clip(color, 0, 1)
                image[y, x] = color
        
        return (image * 255).astype(np.uint8)
    
    else:
        # Use gsplat for proper rendering
        H, W = image_size
        
        # Convert to tensors and flatten H, W to N
        means_t = torch.from_numpy(means).float().to(device).reshape(-1, 3)
        scales_t = torch.from_numpy(scales).float().to(device).reshape(-1, 3)
        rotations_t = torch.from_numpy(rotations).float().to(device).reshape(-1, 4)
        opacities_t = torch.from_numpy(opacities).float().to(device).reshape(-1)
        sh_t = torch.from_numpy(sh).float().to(device).reshape(-1, sh.shape[-1])
        
        # Camera matrix
        K = torch.from_numpy(intrinsic).float().to(device)
        if extrinsic.shape[0] == 4:
            c2w = torch.from_numpy(extrinsic).float().to(device)
        else:
            c2w = torch.eye(4, device=device)
            c2w[:3] = torch.from_numpy(extrinsic).float()
        
        w2c = torch.linalg.inv(c2w)
        
        # gsplat expects quats wxyz, opacities [N], colors [N, 3]
        if rotations_t.shape[-1] == 4:
            quats_t = torch.cat([rotations_t[:, 3:4], rotations_t[:, 0:3]], dim=-1)
        else:
            quats_t = rotations_t

        if opacities_t.ndim > 1:
            opacities_t = opacities_t.squeeze(-1)

        if sh_t.ndim > 2:
            sh_t = sh_t.reshape(-1, sh_t.shape[-1])
        
        # Convert SH DC component to RGB colors since we are not using sh_degree in rasterization
        C0 = 0.28209479177387814
        if sh_t.shape[-1] == 1:
            colors_t = sh_t.repeat(1, 3) * C0 + 0.5
        elif sh_t.shape[-1] >= 3:
            colors_t = sh_t[:, :3] * C0 + 0.5
        else:
            colors_t = torch.zeros((means_t.shape[0], 3), device=device, dtype=means_t.dtype)

        # gsplat always returns (Tensor[C,H,W,X], Tensor[C,H,W,1], Dict)
        # For 'RGB+D', X is 4 (RGB + Depth)
        rendered_t, _alpha, _meta = rasterization(
            means=means_t,
            quats=quats_t,
            scales=scales_t,
            opacities=opacities_t,
            colors=colors_t,
            viewmats=w2c.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=int(W),
            height=int(H),
            render_mode='RGB+D',
        )

        # rendered_t: [C, H, W, 4] with C=1
        rendered_np = rendered_t[0].cpu().numpy()
        rgb = (rendered_np[..., :3].clip(0, 1) * 255).astype(np.uint8)
        depth = rendered_np[..., 3]
        return rgb, depth


def get_reconstructed_scene_nview(
    dataset,
    model: torch.nn.Module,
    retrieval_model: Optional[str],
    device: str,
    output_dir: str,
    photometric_loss_w: float = 0.5,
    matching_conf_thr: float = 5.0,
    random_batch: bool = False,
    max_batches: Optional[int] = None,
    use_gt_poses: bool = False,
    opt_depth: bool = True,
    opt_pose: bool = True,
    opt_focal: bool = True,
) -> List[Dict]:
    """Run N-view reconstruction with optional retrieval and mesh conditioning."""
    
    scenes = []
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=smplx_collate_fn,
        shuffle=random_batch,
    )
    
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="N-View Mesh-Aware SfM")):
        scene_name = batch['scene'][0]
        context_images = batch['context']
        
        print(f"\n[{batch_idx}] {scene_name} with {len(context_images)} views + mesh conditioning")
        
        # Build image list and prepare mesh anchors
        file_paths = []
        for view in context_images:
            img_path = view['image_path'][0] if isinstance(view['image_path'], list) else view['image_path']
            file_paths.append(img_path)

        # Prepare mesh anchors dict keyed by file path (matches img['instance'] in forward_mast3r)
        mesh_anchors_dict = prepare_mesh_anchors_dict(batch, file_paths)
        
        # Load images
        print(f"  Loading {len(file_paths)} images...")
        imgs = load_images(file_paths, size=512)
        
        # Optional retrieval
        sim_matrix = None
        if retrieval_model and HAS_RETRIEVAL:
            print(f"  Running retrieval-based pairing...")
            try:
                retriever = Retriever(retrieval_model, device=device)
                with torch.no_grad():
                    sim_matrix = retriever(file_paths)
                del retriever
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ⚠ Retrieval failed: {e}")
                sim_matrix = None
        
        # Make pairs
        if sim_matrix is not None:
            scene_graph = 'retrieval'
            pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
        else:
            pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
        
        print(f"  Scene graph: {len(pairs)} pairs")
        
        # Run sparse SfM with mesh-aware GCN conditioning
        if use_gt_poses:
            print(f"  [DEBUG MODE] Using Ground Truth camera poses...")
            # Extract GT poses and intrinsics from batch
            gt_c2w = []
            gt_K = []
            for view in context_images:
                gt_c2w.append(view['camera_pose'][0].to(device))
                gt_K.append(view['camera_intrinsics'][0].to(device))
            
            # Still need to run forward pass to get Gaussians in cache
            print(f"  Running forward pass with GCN mesh conditioning...")
            from mast3r.cloud_opt.sparse_ga import convert_dust3r_pairs_naming
            pairs = convert_dust3r_pairs_naming(file_paths, pairs)
            _res_paths, _ = forward_mast3r(pairs, model, cache_path=cache_dir, device=device, mesh_anchors=mesh_anchors_dict)
            
            # Create a mock results dict that mirrors SparseGA structure
            from src.mast3r_src.mast3r.cloud_opt.sparse_ga import make_dense_pts3d
            scene_graph_mock = type('MockSparseGA', (), {
                'get_im_poses': lambda self: torch.stack(gt_c2w),
                'get_dense_pts3d': lambda self: make_dense_pts3d(
                    torch.stack(gt_K), torch.stack(gt_c2w), 
                    [torch.zeros((512, 512), device=device) for _ in gt_c2w], # depthmaps not used if we load canon_attrs
                    [str(p) for p in file_paths], subsample=8, device=device
                ),
                'intrinsics': gt_K,
                'cache_dir': cache_dir # For finding gaussian attributes in exporters
            })()
            scene_graph = scene_graph_mock
        else:
            print(f"  Running sparse SfM with GCN mesh conditioning...")
            scene_graph = sparse_global_alignment(
                file_paths, pairs, cache_dir,
                model,
                lr1=0.07, niter1=500, lr2=0.01, niter2=300,
                device=device,
                opt_depth=opt_depth,
                opt_pose=opt_pose,
                opt_focal=opt_focal,
                matching_conf_thr=matching_conf_thr,
                photometric_loss_w=photometric_loss_w,
                mesh_anchors=mesh_anchors_dict,
            )

        # Wrap scene graph and metadata in a dict
        scene = {
            'graph': scene_graph,
            'scene_name': scene_name,
            'cache_dir': cache_dir,
            'file_paths': file_paths,
            'mesh_anchors': batch['mesh_anchors'][0].cpu().numpy() if 'mesh_anchors' in batch else None,
        }

        scenes.append(scene)

        if max_batches is not None and len(scenes) >= max_batches:
            break
    
    return scenes


def _resolve_dataset_subroot(dataset_root: str, split: Optional[str], names: List[str]) -> str:
    """Resolve dataset subroot with optional split support."""
    for name in names:
        base = os.path.join(dataset_root, name)
        if os.path.isdir(base):
            if split:
                split_base = os.path.join(base, split)
                if os.path.isdir(split_base):
                    return split_base
            return base

    if split:
        split_root = os.path.join(dataset_root, split)
        if os.path.isdir(split_root):
            return split_root

    if os.path.isdir(dataset_root):
        return dataset_root

    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")


def export_gaussian_renderings(
    scene: Dict,
    imgs: List[Dict],
    output_dir: str,
    device: str,
):
    """Render and export gaussian splatting images for all views."""
    
    scene_dir = Path(output_dir) / scene['scene_name']
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Get scene graph and basic info
    scene_graph = scene['graph']
    poses = scene_graph.get_im_poses()  # [V, 3, 4]
    
    print(f"\n  Exporting gaussian renderings for {len(poses)} views...")

    # Get intrinsics from scene graph
    K_list = scene_graph.intrinsics if hasattr(scene_graph, 'intrinsics') else None
    
    # Get image paths and instances for gaussian lookup
    file_paths = scene['file_paths']
    all_instances = [str(p) for p in file_paths]
    
    rendered_count = 0
    
    # Initialize LPIPS
    try:
        import lpips
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lpips_fn = lpips.LPIPS(net='vgg', version='0.1').to(device)
            lpips_fn.eval()
    except ImportError:
        lpips_fn = None
        print("  ⚠ `lpips` package not found, skipping LPIPS metric")
        
    metrics = {'psnr': [], 'lpips': []}
    
    for view_idx in range(len(poses)):
        try:
            pose = poses[view_idx]  # [3, 4]
            img_obj = imgs[view_idx]
            
            # Get image instance for gaussian lookup
            # SparseGA cache uses original file path strings as instances.
            img_instance = str(file_paths[view_idx])
            
            # Find gaussian attributes directly from hashed forward cache.
            gauss_attrs = _find_gaussian_attrs_for_view(scene['cache_dir'], img_instance, all_instances)
            
            if gauss_attrs is None:
                print(f"    ⚠ No gaussian attributes for view {view_idx} ({img_instance}), using white background")
                # Create white background image as fallback
                H, W = _resolve_image_size(img_obj, (512, 512))
                rendered = np.ones((H, W, 3), dtype=np.uint8) * 255
            else:
                # Extract gaussian parameters
                means = gauss_attrs.get('means', None)  # [H, W, 3]
                scales = gauss_attrs.get('scales', None)  # [H, W, 3]
                rotations = gauss_attrs.get('rotations', None)  # [H, W, 4]
                opacities = gauss_attrs.get('opacities', None)  # [H, W, 1]
                sh = gauss_attrs.get('sh', None)  # [H, W, 3*sh_degree]
                
                if means is None or sh is None:
                    print(f"    ⚠ Invalid gaussian attributes for view {view_idx}, skipping")
                    H, W = _resolve_image_size(img_obj, (512, 512))
                    rendered = np.ones((H, W, 3), dtype=np.uint8) * 128
                else:
                    # Convert all to numpy
                    means_np = means.cpu().numpy() if isinstance(means, torch.Tensor) else means
                    scales_np = scales.cpu().numpy() if isinstance(scales, torch.Tensor) else scales
                    rotations_np = rotations.cpu().numpy() if isinstance(rotations, torch.Tensor) else rotations
                    opacities_np = opacities.cpu().numpy() if isinstance(opacities, torch.Tensor) else opacities
                    sh_np = sh.cpu().numpy() if isinstance(sh, torch.Tensor) else sh
                    
                    if sh_np.ndim > 3 and sh_np.shape[-1] == 1:
                        sh_np = sh_np.squeeze(-1)
                    
                    # Add base image SH to residual SH
                    try:
                        img_rgb = img_obj['img'][0].permute(1, 2, 0).cpu().numpy()
                        img_rgb = (img_rgb * 0.5 + 0.5).clip(0, 1)
                        if img_rgb.shape[:2] != means_np.shape[:2]:
                            import cv2
                            img_rgb = cv2.resize(img_rgb, (means_np.shape[1], means_np.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                        base_sh = (img_rgb - 0.5) / 0.28209479177387814
                        sh_np = sh_np + base_sh
                    except Exception as e:
                        print(f"    ⚠ Could not add base SH: {e}")
                    
                    # Get optimized camera parameters from scene_graph
                    refined_poses = scene_graph.get_im_poses()
                    refined_Ks = getattr(scene_graph, 'intrinsics', None)
                    
                    pose = refined_poses[view_idx]
                    pose_np = pose.cpu().numpy() if isinstance(pose, torch.Tensor) else pose
                    
                    if refined_Ks is not None:
                        K_np = refined_Ks[view_idx]
                        if isinstance(K_np, torch.Tensor): K_np = K_np.cpu().numpy()
                    else:
                        # Fallback if intrinsics not in scene_graph (unlikely)
                        H, W = _resolve_image_size(img_obj, (512, 512))
                        K_np = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float32)
                    
                    # Get image size
                    image_size = _resolve_image_size(img_obj, (512, 512))
                    
                    # Render with optimized parameters
                    rendered, depth = render_gaussian_splats(
                        means_np, scales_np, rotations_np, opacities_np, sh_np,
                        K_np, pose_np, image_size, device
                    )
            
            # Prepare ground truth for side-by-side
            try:
                img_rgb = img_obj['img'][0].permute(1, 2, 0).cpu().numpy()
                img_rgb = (img_rgb * 0.5 + 0.5).clip(0, 1)
                H, W = rendered.shape[:2]
                if img_rgb.shape[:2] != (H, W):
                    import cv2
                    img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LANCZOS4)
                gt_img = (img_rgb * 255).astype(np.uint8)
                
                # Prepare Depth Visualization
                try:
                    import cv2
                    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_vis = (depth_norm * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
                except Exception:
                    # Fallback to grayscale if cv2 colormap fails
                    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_vis = (depth_norm * 255).astype(np.uint8)
                    depth_color = np.repeat(depth_vis[:, :, None], 3, axis=-1)
                
                # Calculate metrics
                mse = np.mean((img_rgb - (rendered / 255.0)) ** 2)
                psnr = float(-10 * np.log10(mse) if mse > 0 else 100)
                metrics['psnr'].append(psnr)
                metrics_str = f"PSNR: {psnr:.2f}"
                
                if lpips_fn is not None:
                    with torch.no_grad():
                        gt_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
                        pred_tensor = torch.from_numpy(rendered / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
                        lpips_val = lpips_fn(gt_tensor, pred_tensor).item()
                        metrics['lpips'].append(lpips_val)
                        metrics_str += f" | LPIPS: {lpips_val:.4f}"
                
                print(f"    View {view_idx} metrics: {metrics_str}")
                
                # Concatenate side-by-side (GT, Render, Depth)
                side_by_side = np.concatenate([gt_img, rendered, depth_color], axis=1)
            except Exception as e:
                print(f"    ⚠ Failed to create side-by-side GT or metrics: {e}")
                side_by_side = rendered

            # Save rendered image
            img_path = scene_dir / f'view_{view_idx:03d}_rendered.png'
            Image.fromarray(side_by_side).save(img_path)
            rendered_count += 1
            
        except Exception as e:
            print(f"    ⚠ Error rendering view {view_idx}: {e}")
            continue
    
    print(f"  ✓ Rendered {rendered_count}/{len(poses)} views")
    
    # Save summary
    summary = {
        'scene_name': scene['scene_name'],
        'num_views': len(poses),
        'rendered_views': rendered_count,
        'num_cameras': len(scene['file_paths']),
        'has_mesh_anchors': scene['mesh_anchors'] is not None,
        'avg_psnr': float(np.mean(metrics['psnr'])) if metrics['psnr'] else None,
        'avg_lpips': float(np.mean(metrics['lpips'])) if metrics['lpips'] else None,
    }
    
    summary_path = scene_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ Saved summary to {summary_path}")


def export_scene_ply(scene: Dict, imgs: List[Dict], output_dir: str, conf_thresh: float = 0.0):
    """Export a scene-level gaussian point cloud to `scene.ply`."""
    scene_dir = Path(output_dir) / scene['scene_name']
    scene_dir.mkdir(parents=True, exist_ok=True)

    scene_graph = scene['graph']
    file_paths = scene['file_paths']
    all_instances = [str(p) for p in file_paths]

    if not hasattr(scene_graph, 'get_dense_pts3d'):
        print("  ⚠ SparseGA object has no get_dense_pts3d(); skipping PLY export")
        return

    try:
        dense_pts3d, _, confs = scene_graph.get_dense_pts3d()
        poses = scene_graph.get_im_poses()
    except Exception as e:
        print(f"  ⚠ Failed to get dense points or poses for PLY export: {e}")
        return

    all_xyz, all_op, all_sc, all_rot, all_sh = [], [], [], [], []

    for view_idx in range(min(len(file_paths), len(dense_pts3d))):
        img_instance = str(file_paths[view_idx])
        gauss_attrs = _find_gaussian_attrs_for_view(scene['cache_dir'], img_instance, all_instances)
        if gauss_attrs is None:
            continue

        means = dense_pts3d[view_idx].reshape(-1, 3)
        n = means.shape[0]

        def _as_np(name, default):
            val = gauss_attrs.get(name, None)
            if val is None:
                return default
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            val = np.asarray(val)
            if val.ndim > 2:
                val = val.reshape(-1, val.shape[-1])
            if val.shape[0] < n:
                pad = np.repeat(val[-1:, :], n - val.shape[0], axis=0)
                val = np.concatenate([val, pad], axis=0)
            elif val.shape[0] > n:
                val = val[:n]
            return val

        means_np = means.detach().cpu().numpy() if isinstance(means, torch.Tensor) else np.asarray(means)
        op = _as_np('opacities', np.ones((n, 1), dtype=np.float32) * 2.0)
        sc = _as_np('scales', np.ones((n, 3), dtype=np.float32) * 0.01)
        
        # Rotations from network are xyzw and in local camera coordinates!
        rt_local = _as_np('rotations', np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))) # default xyzw identity
        
        # Transform rotations to world coordinates using cam2w
        try:
            from scipy.spatial.transform import Rotation
            pose = poses[view_idx].cpu().numpy() if isinstance(poses[view_idx], torch.Tensor) else poses[view_idx]
            R_cam2w = pose[:3, :3]
            r_cam = Rotation.from_matrix(R_cam2w)
            r_local = Rotation.from_quat(rt_local) # expects xyzw
            r_global = r_cam * r_local
            rt_global_xyzw = r_global.as_quat().astype(np.float32)
            
            # Convert xyzw -> wxyz for standard 3DGS PLY export
            rt_ply = np.concatenate([rt_global_xyzw[:, 3:4], rt_global_xyzw[:, 0:3]], axis=-1)
        except Exception as e:
            print(f"    ⚠ Failed to rotate quaternions: {e}")
            rt_ply = np.concatenate([rt_local[:, 3:4], rt_local[:, 0:3]], axis=-1)
        
        # Load and combine SH
        sh_res = _as_np('sh', np.zeros((n, 3), dtype=np.float32))
        try:
            img_rgb = imgs[view_idx]['img'][0].permute(1, 2, 0).cpu().numpy()
            img_rgb = (img_rgb * 0.5 + 0.5).clip(0, 1)
            # Find closest integer shape for subsampling if needed
            orig_H, orig_W = img_rgb.shape[:2]
            step = max(1, int(np.sqrt(orig_H * orig_W / n)))
            base_sh = (img_rgb[::step, ::step].reshape(-1, 3)[:n] - 0.5) / 0.28209479177387814
            if base_sh.shape[0] < n:
                pad = np.repeat(base_sh[-1:], n - base_sh.shape[0], axis=0)
                base_sh = np.concatenate([base_sh, pad], axis=0)
            sh = sh_res + base_sh
        except Exception:
            sh = sh_res

        cf = confs[view_idx]
        if isinstance(cf, torch.Tensor):
            cf = cf.detach().cpu().numpy()
        cf = np.asarray(cf).reshape(-1)
        if cf.shape[0] != n:
            # If confidence layout differs, skip filtering for this view.
            mask = np.ones((n,), dtype=bool)
        else:
            mask = cf > conf_thresh if conf_thresh > 0 else np.ones((n,), dtype=bool)

        all_xyz.append(means_np[mask])
        all_op.append(op[mask])
        all_sc.append(sc[mask])
        all_rot.append(rt_ply[mask])
        all_sh.append(sh[mask])

    if not all_xyz:
        print("  ⚠ No gaussian attributes found for any view; skipping scene.ply export")
        return

    xyz = np.concatenate(all_xyz, axis=0)
    op = np.concatenate(all_op, axis=0)
    sc = np.concatenate(all_sc, axis=0)
    rt = np.concatenate(all_rot, axis=0)
    sh = np.concatenate(all_sh, axis=0)

    ply_path = str(scene_dir / 'scene.ply')
    _save_simple_splat_ply(ply_path, xyz, op, sc, rt, sh)
    print(f"  ✓ Saved scene ply: {ply_path} (points={xyz.shape[0]})")


def parse_args():
    parser = argparse.ArgumentParser(description='N-View Mesh-Aware SfM with GCN')

    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--adj_path', type=str, default=None)
    parser.add_argument('--fps_indices_path', type=str, default=None)
    parser.add_argument('--smplx_model_path', type=str, default=None)
    
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--retrieval_ckpt', type=str, default=None)
    
    parser.add_argument('--sequence_id', type=str, default=None, help='Deterministic sequence to run.')
    parser.add_argument('--view_ids', type=str, nargs='+', default=None, help='Deterministic views to run (e.g., cam_00 cam_01).')
    
    parser.add_argument('--num_views', type=int, default=None, help='4-5 views')
    parser.add_argument('--photometric_loss_w', type=float, default=0.5)
    parser.add_argument('--opt_depth', type=str, default='False', help='Whether to optimize depthmaps (True/False)')
    parser.add_argument('--opt_pose', type=str, default='False', help='Whether to optimize camera poses (True/False)')
    parser.add_argument('--opt_focals', type=str, default='False', help='Whether to optimize camera focal lengths (True/False)')
    parser.add_argument(
        '--random_batch',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Shuffle and sample random batches.',
    )
    parser.add_argument('--max_batches', type=int, default=None, help='Max number of batches/scenes to run.')
    
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument(
        '--use_gt_poses',
        action='store_true',
        help='Use Ground Truth camera poses from dataset instead of SfM optimization.'
    )
    args = parser.parse_args()

    defaults = {
        'split': None,
        'num_views': 4,
        'photometric_loss_w': 0.5,
        'output_dir': './results_nview_gcn',
        'max_batches': 1,
        'random_batch': True,
    }

    if args.config:
        if OmegaConf is None:
            raise ImportError("omegaconf is required for --config support. Install with: pip install omegaconf")
        cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config file must contain a mapping at top level: {args.config}")
        for key in [
            'dataset_root', 'split', 'adj_path', 'fps_indices_path', 'smplx_model_path',
            'model', 'retrieval_ckpt', 'num_views', 'photometric_loss_w', 'output_dir',
            'random_batch', 'max_batches', 'sequence_id', 'view_ids',
            'opt_depth', 'opt_pose', 'opt_focals'
        ]:
            current_val = getattr(args, key)
            if current_val is None:
                cfg_val = cfg.get(key, None)
                if cfg_val is not None and cfg_val != '':
                    setattr(args, key, cfg_val)

    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.retrieval_ckpt == '':
        args.retrieval_ckpt = None

    if args.split in ('', 'null', 'None'):
        args.split = None

    required_fields = ['dataset_root', 'adj_path', 'fps_indices_path', 'smplx_model_path', 'model']
    missing = [k for k in required_fields if getattr(args, k) in (None, '')]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")
    
    return args


def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading inference model...")
    cfg_obj = None
    if args.config:
        try:
            cfg_obj = workspace.load_config(args.config, [])
            print(f"✓ Loaded config from {args.config}")
        except Exception as e:
            print(f"⚠ Could not load config {args.config}: {e}")
            cfg_obj = None

    model = load_model_for_inference(args.model, device=device, config=cfg_obj)
    model.eval()
    print("✓ Model loaded\n")
    
    # Check for GCN encoder
    has_gcn = hasattr(model, 'gcn_encoder')
    print(f"GCN Encoder: {'✓ Available' if has_gcn else '✗ Not found'}\n")
    
    # Create dataset
    print("Creating N-view dataset...")
    main_root = _resolve_dataset_subroot(args.dataset_root, args.split, ['main'])
    depth_root = _resolve_dataset_subroot(args.dataset_root, args.split, ['depth', 'depths'])
    anchor_root = _resolve_dataset_subroot(args.dataset_root, args.split, ['anchors', 'anchor'])
    print(f"Resolved roots: main={main_root}, depth={depth_root}, anchors={anchor_root}")

    # Ensure resolution is a tuple (H, W)
    res = 512
    resolution = (res, res) if isinstance(res, int) else res

    dataset = get_anatomical_nview_dataset(
        main_root=main_root,
        depth_root=depth_root,
        anchor_root=anchor_root,
        adj_path=args.adj_path,
        fps_indices_path=args.fps_indices_path,
        smplx_model_path=args.smplx_model_path,
        resolution=resolution,
        num_views=args.num_views,
        sequences=[args.sequence_id] if args.sequence_id else None,
        view_ids=args.view_ids,
    )
    print(f"✓ Dataset: {len(dataset)} samples\n")
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run reconstruction
    print("Starting N-view mesh-aware inference...\n")
    scenes = get_reconstructed_scene_nview(
        dataset=dataset,
        model=model,
        retrieval_model=args.retrieval_ckpt,
        device=device,
        output_dir=args.output_dir,
        photometric_loss_w=args.photometric_loss_w,
        random_batch=args.random_batch,
        max_batches=args.max_batches,
        use_gt_poses=args.use_gt_poses,
        opt_depth=args.opt_depth.lower() == 'true',
        opt_pose=args.opt_pose.lower() == 'true',
        opt_focal=args.opt_focals.lower() == 'true',
    )
    
    # Export renderings
    print("\nExporting gaussian splat renderings...")
    for scene_idx, scene in enumerate(scenes):
        file_paths = scene['file_paths']
        imgs = load_images(file_paths, size=512)
        export_gaussian_renderings(scene, imgs, args.output_dir, device)
        export_scene_ply(scene, imgs, args.output_dir, conf_thresh=0.0)
    
    print(f"\n✓ Complete! Results: {args.output_dir}")

if __name__ == '__main__':
    main()