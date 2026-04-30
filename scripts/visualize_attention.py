"""
scripts/visualize_attention.py

Loads a finetuned MAST3RAnatomicalRefinement checkpoint and visualizes
the prior_attention maps from GaussianHead — showing which image patches
attend to which SMPL-X body anchors.

Usage:
    python scripts/visualize_attention.py \
        configs/finetune_anatomical_mvhumannet.yaml \
        refined_checkpoint_path=/ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=xx.ckpt \
        eval_save_dir=/ssd_scratch/gnrs/attn_viz
"""

import os
import sys

# Add project root to sys.path so 'workspace' and 'train_anatomical_refinement' can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(os.path.join(PROJECT_ROOT, 'src/pixelsplat_src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src/mast3r_src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src/mast3r_src/dust3r'))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import workspace
from train_anatomical_refinement import MAST3RAnatomicalRefinement
from src.anatomical_prior.anatomical_dataset import (
    MVHumanNetAnatomicalData, MVHumanNetAnatomicalDataset,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Attention hook
# ---------------------------------------------------------------------------

class AttentionCapture:
    """
    Register as a forward hook on nn.MultiheadAttention.
    Captures per-head attention weights [B, nH, S_q, S_k].
    """
    def __init__(self):
        self.weights = None   # [B, nH, S_q=num_patches, S_k=512]
        self._handle = None

    def register(self, mha_module):
        def hook(module, inputs, outputs):
            # outputs = (attn_output, attn_weights | None)
            # attn_weights is None unless need_weights=True
            # We monkey-patch forward to force need_weights below.
            pass

        # Monkey-patch the forward to capture weights.
        # nn.MultiheadAttention.forward signature:
        #   forward(query, key, value, ..., need_weights=True, average_attn_weights=True)
        original_forward = mha_module.forward

        def patched_forward(query, key, value, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False  # keep per-head: [B, nH, S_q, S_k]
            out, weights = original_forward(query, key, value, **kwargs)
            self.weights = weights.detach()   # [B, nH, S_q, S_k]
            return out, weights

        mha_module.forward = patched_forward
        self._original_forward = original_forward
        self._module = mha_module

    def restore(self):
        if self._module is not None:
            self._module.forward = self._original_forward


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def attn_to_heatmap(attn_weights, H, W, patch_size=16, head_idx=None):
    """
    attn_weights: [B, nH, S_q, S_k]  — patch→anchor attention
    Returns [B, H, W] heatmap (max over anchors, optionally one head).
    """
    B, nH, S_q, S_k = attn_weights.shape
    if head_idx is not None:
        a = attn_weights[:, head_idx]           # [B, S_q, S_k]
    else:
        a = attn_weights.mean(dim=1)            # avg heads [B, S_q, S_k]

    # For each patch: max attention weight across all 512 anchors
    patch_attn, _ = a.max(dim=-1)              # [B, S_q]

    H_p = H // patch_size
    W_p = W // patch_size
    patch_map = patch_attn.view(B, 1, H_p, W_p)
    heatmap = F.interpolate(patch_map, size=(H, W), mode='bilinear', align_corners=False)
    return heatmap.squeeze(1)                  # [B, H, W]


def anchor_attention_to_image(attn_weights, anchors_world, camera_pose, camera_intrinsics, H, W, patch_size=16):
    """
    Project 512 SMPL-X anchors into the image plane and color each by
    the total attention it received (sum over all query patches).

    Returns an RGB image [H, W, 3] as uint8.
    """
    B, nH, S_q, S_k = attn_weights.shape
    # Sum over query patches → anchor salience [B, 512]
    anchor_sal = attn_weights.mean(dim=1).sum(dim=1)  # [B, 512]
    anchor_sal = anchor_sal[0].cpu().numpy()           # [512]
    # Normalize
    anchor_sal = (anchor_sal - anchor_sal.min()) / (anchor_sal.max() - anchor_sal.min() + 1e-8)

    # Project anchors into the first target view camera
    # anchors_world: [512, 3] in MASt3R canonical space (context[0] cam frame)
    anchors = anchors_world.cpu().numpy()   # [512, 3]
    K = camera_intrinsics[0].cpu().numpy()  # [3, 3]
    # Simple pinhole projection (anchors already in view-space):
    # This is approximate — anchors are in SMPL-X world space, not camera space.
    # For a proper projection you'd apply the camera extrinsics here.
    uvs = anchors[:, :2] / anchors[:, 2:3].clip(min=1e-3)
    uvs[:, 0] = uvs[:, 0] * K[0, 0] + K[0, 2]
    uvs[:, 1] = uvs[:, 1] * K[1, 1] + K[1, 2]

    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cmap = cm.get_cmap('RdYlGn')
    for i, (u, v) in enumerate(uvs):
        u_i, v_i = int(round(u)), int(round(v))
        if 0 <= u_i < W and 0 <= v_i < H:
            color = cmap(anchor_sal[i])[:3]
            r = max(3, int(anchor_sal[i] * 6))
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx**2 + dy**2 <= r**2:
                        yy, xx = v_i + dy, u_i + dx
                        if 0 <= yy < H and 0 <= xx < W:
                            canvas[yy, xx] = color

    return (canvas * 255).astype(np.uint8)


def save_attention_grid(rgb_img, heatmaps_per_head, avg_heatmap, human_mask,
                        sample_key, save_dir, step):
    """
    heatmaps_per_head: list of [H, W] tensors, one per attention head
    avg_heatmap: [H, W]
    """
    nH = len(heatmaps_per_head)
    ncols = 3 + nH  # rgb | human_mask | avg_attn | head0..headN-1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    def overlay(ax, img_chw, heatmap_hw, title, alpha=0.55):
        img_np = img_chw.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        heat_np = heatmap_hw.cpu().numpy()
        ax.imshow(img_np)
        ax.imshow(heat_np, cmap='jet', alpha=alpha, vmin=0, vmax=heat_np.max() + 1e-8)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    img_chw = rgb_img[0]  # [3, H, W]
    axes[0].imshow(img_chw.permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[0].set_title('Input RGB', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(human_mask[0].cpu().numpy(), cmap='gray')
    axes[1].set_title('Human Mask', fontsize=9)
    axes[1].axis('off')

    overlay(axes[2], img_chw, avg_heatmap[0], 'Avg Attention (all heads)')

    for h_idx, hm in enumerate(heatmaps_per_head):
        overlay(axes[3 + h_idx], img_chw, hm[0], f'Head {h_idx}')

    plt.suptitle(f'{sample_key}', fontsize=10)
    plt.tight_layout()

    out_dir = os.path.join(save_dir, 'attention_maps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'attn_{step:04d}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved attention map → {out_path}')
    return out_path


def correlation_with_mask(heatmap_hw, mask_hw):
    """Pearson correlation between attention heatmap and binary human mask."""
    h = heatmap_hw.cpu().float().flatten()
    m = mask_hw.cpu().float().flatten()
    h = h - h.mean()
    m = m - m.mean()
    denom = (h.norm() * m.norm()).clamp(min=1e-8)
    return (h @ m / denom).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_dataset(config):
    dataset_root = getattr(config.data, 'root', '/ssd_scratch/gnrs/mvhumannet++_10demo')
    main_root   = os.path.join(dataset_root, 'main')
    depth_root  = os.path.join(dataset_root, 'depth')
    anchor_root = os.path.join(dataset_root, 'anchors')
    adj_path    = os.path.join(dataset_root, 'smpl_adj_512.npy')
    fps_path    = os.path.join(dataset_root, 'fps_indices_512.npy')
    smplx_model_path = getattr(config.data, 'smplx_model_path', 'smplx_models/')

    data = MVHumanNetAnatomicalData(
        main_root=main_root,
        depth_root=depth_root,
        anchor_root=anchor_root,
        smplx_model_path=smplx_model_path,
        fps_indices_path=fps_path,
        sequences=getattr(config.data, 'sequences', None),
    )
    adj = torch.from_numpy(np.load(adj_path).astype(np.float32))
    return MVHumanNetAnatomicalDataset(
        data=data, adj=adj,
        resolution=config.data.resolution,
        num_epochs_per_epoch=1,
    )


def run_attention_viz(config):
    save_dir = getattr(config, 'eval_save_dir', '/ssd_scratch/gnrs/attn_viz')
    os.makedirs(save_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    refined_ckpt = getattr(config, 'refined_checkpoint_path',
                           getattr(config, 'checkpoint_path', None))
    if refined_ckpt is None:
        raise ValueError('Set config.refined_checkpoint_path to the .ckpt file to inspect.')

    print(f'Loading model from Splatt3R ckpt: {config.splatt3r_checkpoint_path}')
    model = MAST3RAnatomicalRefinement(config)

    print(f'Loading refined weights from: {refined_ckpt}')
    ckpt = torch.load(refined_ckpt, map_location='cpu')
    sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  {len(missing)} missing keys (expected for frozen parts)')

    model.cuda().eval()

    # ── Register attention capture hooks on BOTH heads ───────────────────────
    cap1 = AttentionCapture()
    cap2 = AttentionCapture()
    cap1.register(model.encoder.downstream_head1.prior_attention)
    cap2.register(model.encoder.downstream_head2.prior_attention)

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = _make_dataset(config)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    n_viz = getattr(config, 'num_viz_samples', 10)
    correlations = []

    def to_cuda(obj):
        if torch.is_tensor(obj):
            return obj.cuda()
        elif isinstance(obj, dict):
            return {k: to_cuda(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_cuda(v) for v in obj]
        return obj

    for step, batch in enumerate(loader):
        if step >= n_viz:
            break

        batch = to_cuda(batch)

        view1, view2 = batch['context']
        adj      = batch['smpl_adj'][0]
        anchors1 = view1['mesh_anchors']
        anchors2 = view2['mesh_anchors']

        _, _, H, W = view1['img'].shape

        with torch.no_grad():
            model.forward(view1, view2, adj, anchors1, anchors2)

        # cap1.weights: [B=1, nH=4, S_q=num_patches, S_k=512]
        if cap1.weights is None:
            print(f'[step {step}] No attention weights captured — '
                  'model may not have used human_prior_features.')
            continue

        attn = cap1.weights   # [1, 4, S_q, 512]
        patch_size = model.encoder.downstream_head1.patch_size  # e.g. 16

        # Average heatmap (all heads)
        avg_hm = attn_to_heatmap(attn, H, W, patch_size=patch_size)           # [1, H, W]

        # Per-head heatmaps
        nH = attn.shape[1]
        per_head = [attn_to_heatmap(attn, H, W, patch_size=patch_size, head_idx=h)
                    for h in range(nH)]

        # Human mask from context view 1 (patchified → full res)
        hmask = torch.from_numpy(
            view1['mask_human'][0] if isinstance(view1['mask_human'], np.ndarray)
            else view1['mask_human'][0].cpu().numpy()
        ).float().unsqueeze(0).cuda()   # [1, H, W]

        # Correlation metric
        corr = correlation_with_mask(avg_hm[0], hmask[0])
        correlations.append(corr)

        scene     = batch['scene'][0]
        frame     = batch.get('frame_name', ['unknown'])[0]
        sample_key = f'{scene}/{frame}'

        rgb_img = torch.stack([view1['original_img'][0]], dim=0)  # [1, 3, H, W] — already tensor

        save_attention_grid(
            rgb_img=rgb_img,
            heatmaps_per_head=per_head,
            avg_heatmap=avg_hm,
            human_mask=hmask,
            sample_key=sample_key,
            save_dir=save_dir,
            step=step,
        )

        print(f'  [{step+1:03d}/{n_viz}] {sample_key}  |  attn-mask corr = {corr:.4f}')

    cap1.restore()
    cap2.restore()

    if correlations:
        mean_corr = np.mean(correlations)
        print(f'\nMean attention-mask correlation across {len(correlations)} samples: {mean_corr:.4f}')
        print('  > 0.4  → good localization (attention mostly on human patches)')
        print('  < 0.1  → flat/uniform attention (GCN not localizing yet)')
        with open(os.path.join(save_dir, 'attention_correlation.txt'), 'w') as f:
            f.write(f'mean_correlation: {mean_corr:.6f}\n')
            for i, c in enumerate(correlations):
                f.write(f'sample_{i:04d}: {c:.6f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = workspace.load_config(args.config_path, args.overrides)
    run_attention_viz(config)
