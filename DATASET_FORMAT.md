# MVHumanNet++ Dataset Format for Anatomical Refinement

This document is the canonical reference for preparing the MVHumanNet++ dataset
for the Splatt3R anatomical refinement pipeline (GCN + Cross-Attention).

---

## 0. PEAR Mesh Source — Key Fact

PEAR does **not** output a custom mesh. Internally it uses the `ehm()` function to
convert its body/FLAME parameters into **standard SMPL-X vertices**:

```python
pd_smplx_dict = ehm(outputs['body_param'], outputs['flame_param'], pose_type='aa')
mesh_vertices  = pd_smplx_dict['vertices'][0].detach().cpu().numpy()  # [10475, 3]
save_obj(mesh_path, mesh_vertices, mesh_faces)  # mesh_faces = fixed SMPL-X faces
```

This means:
- **10,475 vertices** — the standard SMPL-X vertex count, always.
- **Fixed topology** — `mesh_faces` is the SMPL-X face array (20,908 triangles).
  It **never changes** across frames, sequences, or subjects.
- **Adjacency matrix is computed ONCE** from the SMPL-X faces + FPS indices.
  It is shared globally with no per-frame or per-sequence variation.

---

## 1. Top-Level Directory Structure

```
MVHumanNet++/
├── main/               ← existing: RGB images, cameras, SMPL-X params
├── depth/              ← existing: EXR depth maps
├── normal/             ← NEW: pre-rendered normal map PNGs
├── anchors/            ← NEW: per-frame 512-anchor NPZs from PEAR
├── fps_indices_512.npy ← NEW: canonical FPS indices (generated ONCE from T-pose)
└── smpl_adj_512.npy    ← NEW: GCN adjacency matrix  (generated ONCE from SMPL-X faces)
```

> **Do NOT restructure `main/` or `depth/`.** Only add the four items above.

---

## 2. Existing Directories (Unchanged)

### `main/{seq}/`
```
main/{seq}/
├── images/{cam_id}/{frame_name}.jpg     # RGB, e.g. cam_00/0025.jpg
├── cameras/{cam_id}/camera.npz          # intrinsics + extrinsics
└── smplx_params/{frame_name}.npz        # per-frame SMPL-X parameters
```

### `depth/{seq}/`
```
depth/{seq}/depths/{cam_id}/{frame_name}.exr   # float32 EXR depth in metres
```

Camera IDs: `cam_00` through `cam_15` (16 cameras, 360° ring).
Frame names: zero-padded integers, e.g. `0025`, `0050`, ..., `1500`.

---

## 3. New Directories

### 3.1 `normal/{seq}/{cam_id}/{frame_name}.png` — Ground-Truth Normal Maps

| Property | Value |
|---|---|
| Format | PNG, uint8 |
| Shape | `(H, W, 3)` – same resolution as corresponding RGB image |
| Channel encoding | R = X,  G = Y,  B = Z |
| Value range | `[0, 255]`  →  pixel = `(normal_xyz * 0.5 + 0.5) * 255` |
| Space | World-space OpenCV convention (Y-down, Z-forward) |

**Conversion in training code:**
```python
normal = (img_uint8 / 255.0) * 2.0 - 1.0   # → [-1, 1]
normal = F.normalize(normal, dim=0)          # unit length per pixel
```

**What it looks like:** Predominantly blue-green-purplish, because:
- Z (blue) dominates on forward-facing body surfaces
- Y (green) appears on top-facing surfaces (shoulders, top of head)
- Background pixels: neutral blue `[128, 128, 255]` → `(0, 0, 1)` in [-1,1]

```
normal/{seq}/
└── cam_00/
│   └── 0025.png
│   └── 0050.png
└── cam_01/
    └── 0025.png
    ...
```

---

### 3.2 `anchors/{seq}/{frame_name}.npz` — SMPL-X Posed Anchor Vertices

| Property | Value |
|---|---|
| Format | NumPy NPZ |
| NPZ key | `anchors` |
| Shape | `(512, 3)` float32 |
| Coordinate space | World space (same as SMPL-X output) |
| Per-frame | Yes — positions change with pose |
| Per-camera | No — shared across all cameras of the same frame |

```
anchors/{seq}/
└── 0025.npz    # np.load(path)['anchors'] → (512, 3)
└── 0050.npz
...
```

> The 512 vertex **indices** are fixed (derived from T-pose FPS). Only the
> vertex **positions** (`xyz` coordinates) change per frame as the body deforms.

---

### 3.3 `fps_indices_512.npy` — Canonical FPS Vertex Indices

| Property | Value |
|---|---|
| Shape | `(512,)` int64 |
| Generated | **Once**, from SMPL-X T-pose/rest-pose |
| Reused | For every sequence, every frame, every camera |

This array maps the reduced set of 512 anchors back to the full 10,475-vertex
SMPL-X mesh. Index `fps_indices[i]` is the global vertex ID of anchor `i`.

**Why T-pose for FPS?**  
Running Farthest Point Sampling on the T-pose guarantees that anchor `i` always
refers to the same anatomical landmark (e.g., left elbow, right knee), regardless
of the body's current pose. If FPS were re-run per-frame, the semantic meaning of
each anchor node would change, making the GCN unable to learn consistent features.

---

### 3.4 `smpl_adj_512.npy` — GCN Adjacency Matrix

| Property | Value |
|---|---|
| Shape | `(512, 512)` float32 |
| Generated | **Once**, from SMPL-X mesh faces + `fps_indices_512.npy` |
| Normalization | Degree normalized: `D^{-1/2} A D^{-1/2}` with self-loops |
| Value range | `[0, 1]` (after normalization) |

`A[i, j] = 1` (before normalization) if anchor `i` and anchor `j` share a mesh
edge in the original SMPL-X topology. Edges where only one endpoint survived
FPS are discarded. Self-loops are added so each node aggregates its own features.

---

## 4. Anchor Generation Pipeline

### Who Runs This?

An **offline preprocessing agent** runs once per dataset, before any training.
It does not run during training. The agent needs:
- PEAR model weights
- SMPL-X model files (`smplx_lib`)
- Access to the MVHumanNet++ `main/` images

---

### Step A: Generate Global Files (Run ONCE for the Entire Dataset)

These two files are **not per-frame**. They encode the fixed SMPL-X topology and
the canonical 512 anchor positions on the T-pose body. They are saved at the
dataset root and reused forever.

```python
import smplx, torch, numpy as np

# 1. Load SMPL-X in rest/T-pose (zero betas, zero pose)
body = smplx.create(smplx_model_path, model_type='smplx', ...)
with torch.no_grad():
    rest = body(return_verts=True)

rest_vertices = rest.vertices[0].numpy()   # [10475, 3]
smplx_faces   = body.faces                 # [20908, 3]  ← fixed topology

# 2. Run FPS on T-pose vertices → 512 canonical indices
fps_indices = farthest_point_sampling(rest_vertices, 512)
np.save('fps_indices_512.npy', fps_indices)

# 3. Build degree-normalised adjacency matrix from SMPL-X faces
#    Only edges where both endpoints are in fps_indices are kept
adj = build_adjacency(fps_indices, smplx_faces)   # [512, 512]
np.save('smpl_adj_512.npy', adj)
```

**Why T-pose for FPS?**
Anchor `i` always refers to the same anatomical landmark (e.g., left elbow)
regardless of frame. If FPS were re-run on a crouching pose, different body
parts would be sampled and the GCN couldn't learn consistent features.

---

### Step B: Per-Frame Anchor Extraction (Run for Every Frame)

For each `(seq, frame_name)`:

```
PEAR inference on images from multiple cameras (e.g. 4-6 views)
        ↓
ehm(outputs['body_param'], outputs['flame_param'], pose_type='aa')
        ↓
pd_smplx_dict['vertices'][0]   → [10475, 3]  float32
        ↓
vertices[fps_indices]           → [512, 3]    float32
        ↓
np.savez('anchors/{seq}/{frame_name}.npz', anchors=anchor_vertices)
```

**Key points:**
- PEAR is run on **multiple camera views simultaneously** (it is a multi-view method).
  It returns ONE mesh per frame even though it consumed N camera images.
- The output is SMPL-X vertices (via `ehm()`), same topology as the T-pose above.
- `fps_indices` from Step A directly indexes into this output — no remapping needed.
- The resulting `[512, 3]` anchors are **shared across ALL cameras of that frame**.
  cam_00, cam_08, cam_13 at frame 0025 all use the same `anchors/seq/0025.npz`.

```
anchors/{seq}/
├── 0025.npz    # np.load(path)['anchors'] → (512, 3)
├── 0050.npz
└── ...
```

---

## 5. Human Mask

**Computed on-the-fly from the depth map — no separate mask file needed:**

```python
mask_human = depth > 0.0   # bool, shape (H, W)
```

Background pixels have `depth == 0` in EXR convention. This mask is:
- Added to context views as `view['mask_human']`  
- Used to gate `valid_mask` (only valid non-background pts are passed to MASt3R)
- Used in training loss as the human-region weighting mask

Optionally, a cached binary PNG mask can be saved at `mask/{seq}/{cam_id}/{frame}.png`
if the depth-derived mask is too noisy.

---

## 6. Camera Sampling Strategy

16 cameras are arranged in a **360° ring** (cam_00..cam_15, roughly evenly spaced).

| Split | Cameras | Count |
|---|---|---|
| Context | Randomly selected, adjacent in the ring | 2 |
| Target | At least 4 ring positions away from any context cam | 1–3 |

**Why angular gap matters:** If context and target cameras are adjacent, the model
can exploit subtle parallax cues rather than learning real view synthesis from the
geometry. A gap of ≥4 positions (~90°) forces genuine novel-view generation.

The dataloader implements this with a "circular distance" check:
```python
circular_dist = min(abs(i - c), NUM_CAMERAS - abs(i - c))
is_far = circular_dist >= 4   # at least ~90°
```

---

## 7. Batch Dict Keys Reference

All keys expected by `train_anatomical_refinement.py`:

### `batch['context'][i]`

| Key | Type | Shape | Description |
|---|---|---|---|
| `img` | Tensor | `[3,H,W]` | ImageNet-normalised, encoder input |
| `original_img` | Tensor | `[3,H,W]` | Raw RGB in `[0,1]`, for SH residual & MSE |
| `depthmap` | ndarray | `[H,W]` | Metric depth in metres |
| `camera_pose` | ndarray | `[4,4]` | C2W, OpenCV convention |
| `camera_intrinsics` | ndarray | `[4,4]` | Camera K matrix |
| `pts3d` | ndarray | `[H,W,3]` | Unprojected depth points |
| `valid_mask` | ndarray bool | `[H,W]` | `depth>0 & finite & human` |
| `mask_human` | ndarray bool | `[H,W]` | `depth > 0` |
| `mesh_anchors` | Tensor | `[512,3]` | Posed SMPL-X anchor vertices |

### `batch['target'][i]`

| Key | Type | Shape | Description |
|---|---|---|---|
| `original_img` | Tensor | `[3,H,W]` | GT RGB in `[0,1]` |
| `camera_pose` | ndarray | `[4,4]` | C2W |
| `camera_intrinsics` | ndarray | `[4,4]` | K matrix |
| `depthmap` | ndarray | `[H,W]` | Metric depth |
| `mask_human` | ndarray bool | `[H,W]` | `depth > 0` |
| `normal_map` | Tensor | `[3,H,W]` | GT normal in `[0,1]` |

### `batch` (top-level)

| Key | Type | Shape | Description |
|---|---|---|---|
| `smpl_adj` | Tensor | `[512,512]` | Fixed GCN adjacency matrix |
| `smplx` | dict | — | Raw SMPL-X params (for MPJPE loss) |

---

## 8. Storage Estimate

| Item | Per-frame Size | 1000 frames × 16 cams |
|---|---|---|
| RGB `.jpg` | ~200 KB | ~3.2 GB |
| Depth `.exr` | ~4 MB | ~64 GB |
| Normal `.png` | ~400 KB | ~6.4 GB |
| Anchors `.npz` | ~6 KB | ~6 MB (shared across cams) |
| Adj + FPS indices | — | ~1 MB (global, once) |

Normals are the dominant new storage cost. If space is limited, generate them on-the-fly
from the SMPL-X mesh via normal estimation (slower but zero storage).
