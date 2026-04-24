# Sharin-GAN: Anatomical Refinement via Cross-Attention — Final Implementation Plan

A human-aware extension of Splatt3R that conditions the Gaussian prediction head
on SMPL-X mesh priors via a dense GCN and Cross-Attention injection.

---

## Architecture Overview

```
RGB View1, View2
      │
      ▼
 MASt3R ViT Encoder + Cross-Attention Decoder   ← FROZEN
      │
      ├─── decout[0]  (encoder tokens [B, N, 1024])
      └─── decout[-1] (decoder tokens [B, N, 768])  ← modified below
                              │
      SMPL-X 512 Anchors [B,512,3]              ← loaded from NPZ offline
              │
              ▼
      AnatomicalGCNEncoder   (3→128→128→32)     ← TRAINABLE
              │
              ▼
      GaussianHead.prior_attention               ← TRAINABLE (zero-init)
      MultiheadAttention(Q=decout[-1], K/V=GCN output)
              │ residual + LayerNorm
              ▼
      gaussian_dpt (DPT)                         ← TRAINABLE (10x lower LR)
              │
              ▼
      Gaussian Params: means, scales, rotations, sh, opacities
              │
              ▼
      DecoderSplattingCUDA  (Differentiable Renderer)  ← NOT trained (no params)
         ├── render_cuda()           → RGB image  [B,V,3,H,W]
         └── render_normals_cuda()  → Normal map  [B,V,3,H,W]
              │
              ▼
      Loss Computation  (in train_anatomical_refinement.py)
         ├── MSE  (soft-masked: human=1.0, bg=0.1)
         ├── LPIPS (human-mask only)
         └── Normal Consistency (human-mask only, cosine similarity)
```

---

## Files Changed

### `src/mast3r_src/mast3r/catmlp_dpt_head.py` [MODIFIED]
- Added `prior_attention` (`nn.MultiheadAttention`, `embed_dim=768, kdim=32, vdim=32, num_heads=4`)
  and `norm` (`nn.LayerNorm(768)`) to `GaussianHead.__init__`.
- **Zero-initialized `prior_attention.out_proj.weight` and `.bias`** so the head starts
  identical to the pretrained checkpoint at step 0 (ReZero trick — no covariate shift).
- `GaussianHead.forward` now accepts optional `human_prior_features=[B,512,32]`.
  If provided, applies cross-attention on the deepest decoder token sequence
  (`decout[-1]`) and adds a residual before feeding into `gaussian_dpt`.

### `src/pixelsplat_src/cuda_splatting.py` [MODIFIED]
- Added `quaternion_to_surface_normal(quaternions)` — extracts local Z axis (shortest
  ellipsoid axis = surface normal) from each Gaussian's quaternion via the 3rd column
  of the rotation matrix.
- Added `render_normals_cuda(...)` — renders a per-pixel surface normal map by:
  1. Computing per-Gaussian normal `[G,3]` from quaternions.
  2. Passing normals as **`colors_precomp`** (bypasses SH evaluation entirely, which
     only makes sense for RGB view-dependent color — meaningless for normals).
  3. Alpha-compositing via the same CUDA rasterizer used for RGB.
  4. L2-normalising the output per pixel.
- **Why fake colors / `colors_precomp`?** The CUDA rasterizer (`diff_gaussian_rasterization`)
  has two modes: `shs=...` (runs the internal SH shader, output is RGB) or
  `colors_precomp=...` (skips SH, uses raw `[G,3]` directly as color). Normals are
  `[G,3]` vectors — we cannot feed them through the SH shader (it would treat them as
  harmonic color coefficients). `colors_precomp` alpha-composites our normals identically
  to how it composites RGB, producing a correct alpha-weighted normal map output.
- The rendered normals will appear green/blue/purplish — identical to GT normal PNGs
  because the encoding is the same: XYZ→RGB remapped from `[-1,1]` to `[0,1]`.
  Z (blue/outward-facing) dominates for forward-facing humans.

### `src/pixelsplat_src/decoder_splatting_cuda.py` [MODIFIED]
- Imported `render_normals_cuda`.
- Added `render_normals(batch, pred1, pred2, image_shape)` method that mirrors
  `forward()` but calls `render_normals_cuda` with the Gaussian quaternions.
  Returns `[B, V, 3, H, W]`.

### `src/anatomical_prior/gcn_encoder.py` [NEW]
- `DenseGCNLayer`: standard `A @ (X @ W) + b` using `torch.matmul` (no `torch_geometric`
  or `PyTorch3D` — they break DDP sparse tensor gathers and require complex CUDA installs).
- `AnatomicalGCNEncoder`: 3-layer dense GCN, `in=3 → hidden=128 → out=32`.
  512 fixed SMPL-X nodes, adjacency matrix broadcast across batch.

### `src/anatomical_prior/anatomical_dataset.py` [NEW — see Dataset Requirements below]
- `AnatomicalRefinementDataset` wraps `DUST3RSplattingDataset`.
- Injects `mesh_anchors [512,3]`, `smpl_adj [512,512]`, and `normal_map [3,H,W]`
  into each context/target view dictionary.

### `train_anatomical_refinement.py` [NEW]
- `MAST3RAnatomicalRefinement` Lightning module.
- **Checkpoint loading**: uses `MAST3RGaussians.load_from_checkpoint()` (Lightning API),
  NOT raw `load_state_dict` on a manually constructed model. This ensures all Gaussian DPT
  weights are loaded under the correct key hierarchy.
- **Two optimizer param groups**: `gcn + attention` at full LR, `gaussian_dpt` at 0.1× LR.
- **`calculate_loss`**:
  - Soft mask: `weight = 1.0` on human pixels, `weight = 0.1` on background.
  - MSE applied over soft mask.
  - LPIPS applied on human pixels only.
  - Normal Consistency: `1 - cosine_similarity(rendered_normals, gt_normals)` on human pixels.
  - Normal map GT is read from `batch['target'][*]['normal_map']`, remapped `[0,1]→[-1,1]`.

### `inference_anatomical_refinement.py` [NEW]
- Zero-shot inference: loads checkpoint, runs GCN + cross-attention forward, renders
  novel view, saves to `inference_result.png`.

---

## Dataset Requirements (for Data Preparation)

### Directory Structure Expected
```
dataset_root/
├── splits/
│   ├── train.txt         # one subject/scene ID per line
│   └── val.txt
├── {subject_id}/
│   ├── {cam_id}/
│   │   ├── rgb/
│   │   │   └── {frame_id}.jpg        # RGB image, any resolution (resized to 512×512)
│   │   ├── normal/
│   │   │   └── {frame_id}.png        # GT Normal map, same resolution as RGB
│   │   │                              # Pixel encoding: R=X, G=Y, B=Z, range [0,255]
│   │   │                              # (standard tangent-space or world-space normals)
│   │   ├── depth/
│   │   │   └── {frame_id}.png        # Depth map, uint16, scale factor 1000 (mm→m)
│   │   └── camera.json               # Intrinsics + extrinsics (see format below)
│   └── smplx/
│       └── {frame_id}_anchors.npz    # Pre-sampled 512 SMPL-X anchor vertices
│                                      # Keys: 'vertices' [512,3], 'adjacency' [512,512]
└── smpl_adj_512.npy                  # Global fixed adjacency matrix [512,512]
                                       # (shared across all subjects)
```

### `camera.json` Format
```json
{
  "fl_x": 1234.5,
  "fl_y": 1234.5,
  "cx": 512.0,
  "cy": 512.0,
  "transform_matrix": [[r00,r01,r02,tx],[r10,...],[r20,...],[0,0,0,1]]
}
```

### Preprocessing Steps (Run Once Offline)
1. **PEAR / SMPL-X fitting**: Run PEAR on every frame to get the full 10,475 SMPL-X mesh vertices.
2. **FPS Sampling**: Farthest Point Sampling from 10,475 → 512 anchor points. Save as `{frame_id}_anchors.npz` with key `'vertices' [512,3]`.
3. **Adjacency Matrix**: Build the 512×512 binary adjacency matrix from SMPL-X topology (edges between subsampled vertices). Save as `smpl_adj_512.npy`. This is computed **once** and reused for all frames/subjects.
4. **Normal Maps**: Convert your renderer output to world-space normals, encode as standard `R=X, G=Y, B=Z` PNG (range 0–255, remapped from [-1,1]).
5. **Depth Maps**: Save as uint16 PNG. Scale = 1000 (value / 1000 = depth in meters). Pixels with depth==0 are treated as background in the mask.
6. **Dataset Index**: Create `train.txt` / `val.txt` listing valid triplets as `{subject_id},{context_cam1},{context_cam2},{target_cam},{frame_id}`.

### What the Dataloader Returns (Per Batch Item)
| Field | Location in Batch | Shape | Description |
|---|---|---|---|
| `img` | `context[i]` | `[3,H,W]` | Preprocessed RGB (normalized, resized) |
| `original_img` | `context[i]` | `[3,H,W]` | Raw RGB in [0,1] (for SH residual) |
| `camera_pose` | `context[i]` | `[4,4]` | C2W transform matrix |
| `camera_intrinsics` | `context[i]` | `[4,4]` | Camera K matrix |
| `depthmap` | `context[i]` | `[H,W]` | Metric depth in meters |
| `mesh_anchors` | `context[i]` | `[512,3]` | SMPL-X 512 anchor vertices |
| `smpl_adj` | `batch` | `[512,512]` | Fixed adjacency matrix (one per batch) |
| `original_img` | `target[i]` | `[3,H,W]` | GT RGB for MSE/LPIPS loss |
| `normal_map` | `target[i]` | `[3,H,W]` | GT normal map in [0,1] |
| `camera_pose` | `target[i]` | `[4,4]` | Target view C2W pose |
| `camera_intrinsics` | `target[i]` | `[4,4]` | Target view intrinsics |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Dense GCN over `torch_geometric` | 512 nodes → dense matmul is faster; torch_geometric sparse ops crash DDP |
| Zero-init `out_proj` in CrossAttn | ReZero trick: model starts identical to pretrained checkpoint at step 0 |
| `colors_precomp` for normal rendering | SH shader is view-dependent color only; `colors_precomp` directly composites `[G,3]` arbitrary values |
| Two LR groups | GCN/attention from scratch (full LR), DPT pretrained (0.1× LR) |
| Soft mask (not binary) | Background weight=0.1 prevents network from hallucinating humans everywhere at inference |
| Erode mask 3px | Avoids penalising boundary pixels where depth map is unreliable |
| Lightning `load_from_checkpoint` | Correct key hierarchy; raw `load_state_dict` on manually-built model silently fails |
