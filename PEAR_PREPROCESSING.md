# PEAR Preprocessing Agent — README

This document tells an agent (or developer) exactly how to modify PEAR's inference
script to produce the anchor NPZ files required by the Splatt3R anatomical refinement
training pipeline.

---

## What You Need to Produce

The training pipeline requires two types of outputs:

| File | Generated | Description |
|---|---|---|
| `fps_indices_512.npy` | **Once, globally** | 512 canonical vertex indices from SMPL-X T-pose |
| `smpl_adj_512.npy` | **Once, globally** | `[512,512]` GCN adjacency matrix |
| `anchors/{seq}/{frame}.npz` | **Once per frame** | `[512,3]` posed SMPL-X anchor vertices |

The anchor file key is `anchors` (not `vertices`). The anchor file is **per-frame,
shared across all cameras** — if a sequence has 16 cameras, you still only produce
one `.npz` per frame.

---

## Understanding the Existing PEAR Output Code

The relevant section of PEAR's inference loop (already in the PEAR codebase) is:

```python
pd_smplx_dict = ehm(outputs['body_param'], outputs['flame_param'], pose_type='aa')
mesh_vertices  = pd_smplx_dict['vertices'][0].detach().cpu().numpy()  # [10475, 3]
mesh_path = os.path.join(mesh_output_path, f"{img_name}_person_{bbox_id:02d}.obj")
save_obj(mesh_path, mesh_vertices, mesh_faces)
```

Key facts about this output:
- `mesh_vertices` shape is always `[10475, 3]` — standard SMPL-X vertex count.
- `mesh_faces` is the fixed SMPL-X triangulation (20,908 faces, never changes).
- The outer loop is typically over images/bounding boxes — it processes one camera
  view at a time, but produces one mesh per person detected.
- `ehm()` converts PEAR's internal representation to standard SMPL-X.

---

## Step 1: Generate Global Files (Run Once Before Processing Frames)

Add this as a standalone function and call it **before** the main frame loop.

```python
import smplx
import numpy as np
import torch
import os


def farthest_point_sampling(points, n_samples):
    """Pure NumPy FPS. Returns selected indices [n_samples]."""
    N = len(points)
    selected = np.zeros(n_samples, dtype=int)
    distances = np.full(N, np.inf)
    selected[0] = np.random.randint(0, N)
    for i in range(1, n_samples):
        last = points[selected[i - 1]]
        dist = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)
    return selected


def build_adjacency(fps_indices, smplx_faces, n_nodes=512):
    """Build D^{-1/2} A D^{-1/2} normalised adjacency from SMPL-X faces."""
    idx_map = {v: i for i, v in enumerate(fps_indices.tolist())}
    fps_set = set(fps_indices.tolist())
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for face in smplx_faces:
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[0], face[2])]:
            if a in fps_set and b in fps_set:
                ia, ib = idx_map[a], idx_map[b]
                adj[ia, ib] = 1.0
                adj[ib, ia] = 1.0
    np.fill_diagonal(adj, 1.0)
    degree = adj.sum(axis=1).clip(min=1)
    adj = adj / np.sqrt(degree[:, None]) / np.sqrt(degree[None, :])
    return adj


def generate_global_files(smplx_model_path, output_dir, n_anchors=512):
    """
    Run once before any per-frame processing.
    Saves fps_indices_512.npy and smpl_adj_512.npy to output_dir.
    """
    fps_path = os.path.join(output_dir, f'fps_indices_{n_anchors}.npy')
    adj_path = os.path.join(output_dir, f'smpl_adj_{n_anchors}.npy')

    if os.path.exists(fps_path) and os.path.exists(adj_path):
        print(f"[SKIP] Global files already exist: {fps_path}, {adj_path}")
        fps_indices = np.load(fps_path).astype(int)
        return fps_indices

    print("Generating global FPS indices and adjacency matrix...")
    body = smplx.create(
        smplx_model_path, model_type='smplx', gender='neutral',
        num_betas=10, num_expression_coeffs=10,
        num_pca_comps=6, use_pca=True, flat_hand_mean=True,
    )
    body.eval()
    with torch.no_grad():
        rest = body(return_verts=True)

    rest_vertices = rest.vertices[0].cpu().numpy()   # [10475, 3]
    smplx_faces   = body.faces                        # [20908, 3]

    fps_indices = farthest_point_sampling(rest_vertices, n_anchors)
    np.save(fps_path, fps_indices)
    print(f"  Saved: {fps_path}")

    adj = build_adjacency(fps_indices, smplx_faces, n_nodes=n_anchors)
    np.save(adj_path, adj)
    print(f"  Saved: {adj_path}  sparsity={1-(adj>0).mean():.1%}")

    return fps_indices
```

---

## Step 2: Track Which Frames Have Been Processed

PEAR loops over individual images. Multiple images from different cameras can belong
to the same frame. You must ensure you only save **one anchor NPZ per frame** (not
one per camera). Add a `processed_frames` set to avoid duplicates.

```python
# Add near the top of the inference script, before the image loop:
processed_frames = set()
```

---

## Step 3: Modify the Per-Image Inference Loop

Find the block that calls `ehm()` and saves the OBJ. Add the anchor extraction
**immediately after** the existing `save_obj` call:

### Existing code (do not remove):
```python
pd_smplx_dict = ehm(outputs['body_param'], outputs['flame_param'], pose_type='aa')
mesh_vertices  = pd_smplx_dict['vertices'][0].detach().cpu().numpy()
mesh_path = os.path.join(mesh_output_path, f"{img_name}_person_{bbox_id:02d}.obj")
save_obj(mesh_path, mesh_vertices, mesh_faces)
```

### Add immediately after (insert this block):
```python
# ── Anchor extraction for Splatt3R anatomical refinement ──────────────────
# The anchor NPZ is per-frame, shared across cameras.
# Parse seq and frame_name from the image filename convention:
#   img_name format: "{seq}_{cam_id}_{frame_name}"  or similar
# Adjust the parsing below to match your actual PEAR img_name format.

seq, cam_id, frame_name = parse_img_name(img_name)  # implement below
frame_key = (seq, frame_name)

if frame_key not in processed_frames:
    anchor_dir = os.path.join(anchor_output_root, seq)
    os.makedirs(anchor_dir, exist_ok=True)
    anchor_path = os.path.join(anchor_dir, f'{frame_name}.npz')

    # Index the 10475 SMPL-X vertices with our fixed 512 FPS indices
    anchor_vertices = mesh_vertices[fps_indices]  # [512, 3] float32
    np.savez(anchor_path, anchors=anchor_vertices)

    processed_frames.add(frame_key)
    print(f"  Saved anchor: {anchor_path}")
# ──────────────────────────────────────────────────────────────────────────
```

---

## Step 4: Implement `parse_img_name` for MVHumanNet++

The image files in MVHumanNet++ follow the pattern:
```
main/{seq}/images/{cam_id}/{frame_name}.jpg
```

When PEAR loads images, `img_name` is typically derived from the file path.
You need to extract `seq`, `cam_id`, and `frame_name` from whatever variable
PEAR uses to identify the current image. Adjust the parser to match:

```python
def parse_img_name(img_name):
    """
    Parse a PEAR img_name string into (seq, cam_id, frame_name).

    MVHumanNet++ example:
      If PEAR receives the full path: main/100831/images/cam_00/0025.jpg
      Then:    seq='100831',  cam_id='cam_00',  frame_name='0025'

    Adjust the split logic to match exactly how PEAR constructs img_name
    in your version of the inference script.
    """
    # Option A: if img_name is the full path
    parts = img_name.replace('\\', '/').split('/')
    # Find 'images' in path: .../{seq}/images/{cam_id}/{frame}.jpg
    try:
        img_idx  = parts.index('images')
        seq      = parts[img_idx - 1]
        cam_id   = parts[img_idx + 1]
        frame_name = os.path.splitext(parts[img_idx + 2])[0]
        return seq, cam_id, frame_name
    except (ValueError, IndexError):
        pass

    # Option B: if img_name is already just the stem "{seq}_{cam_id}_{frame}"
    tokens = img_name.split('_')
    seq, cam_id, frame_name = tokens[0], f'{tokens[1]}_{tokens[2]}', tokens[3]
    return seq, cam_id, frame_name
```

---

## Step 5: Full Invocation Pattern

At the top of the PEAR inference `main()` or wherever it starts:

```python
# ── One-time global setup ─────────────────────────────────────────────────
SMPLX_MODEL_PATH  = '/path/to/smplx_models/'
DATASET_ROOT      = '/path/to/MVHumanNet++/'
ANCHOR_OUTPUT_ROOT = os.path.join(DATASET_ROOT, 'anchors')
N_ANCHORS         = 512

fps_indices = generate_global_files(SMPLX_MODEL_PATH, DATASET_ROOT, N_ANCHORS)
processed_frames = set()
# ──────────────────────────────────────────────────────────────────────────
```

Then run PEAR normally over all images. The anchor files will be written automatically.

---

## Step 6: Verify Output

After running PEAR on a sequence, check:

```bash
# Should exist:
ls MVHumanNet++/fps_indices_512.npy
ls MVHumanNet++/smpl_adj_512.npy

# Should have one NPZ per frame (NOT per camera):
ls MVHumanNet++/anchors/100831/
# → 0025.npz  0050.npz  0075.npz  ...

# Verify NPZ content:
python3 -c "
import numpy as np
d = np.load('MVHumanNet++/anchors/100831/0025.npz')
print(d.files)           # should print: ['anchors']
print(d['anchors'].shape)  # should print: (512, 3)
print(d['anchors'].dtype)  # should print: float32
"
```

---

## Expected File Count

After processing the full dataset:

```
MVHumanNet++/
├── fps_indices_512.npy    ← 1 file
├── smpl_adj_512.npy       ← 1 file
└── anchors/
    ├── 100831/
    │   ├── 0025.npz       ─┐
    │   ├── 0050.npz        │  one per frame (NOT per camera)
    │   └── ...             │  ~60 frames per sequence
    └── 100832/            ─┘
        └── ...
```

If you have 100 sequences × 60 frames = 6,000 NPZ files. Each is ~6KB.
Total additional storage: ~36MB — negligible.

---

## Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---|---|---|
| Saving one NPZ per camera (16×) | Disk waste, training loads wrong file | Use `processed_frames` set |
| NPZ key is `vertices` not `anchors` | Dataloader crashes with KeyError | Use `np.savez(..., anchors=...)` |
| Running FPS per-frame | Anchor semantics change → GCN learns noise | Run FPS once on T-pose |
| Using PEAR vertices directly (10475) without FPS | Dataloader expects [512,3] | Always slice with `fps_indices` |
| Forgetting `smpl_adj_512.npy` | Training crashes at dataset init | Run `generate_global_files()` first |

---

## Note: Global Files Need No Images

`generate_global_files()` only requires:
- The **SMPL-X model NPZ files** (download from https://smpl-x.is.tue.mpg.de/)
- Python packages: `smplx`, `torch`, `numpy`

It does **not** require any human images, any MVHumanNet++ data, or PEAR.
It just loads the SMPL-X model, runs a zero-pose forward pass, and does FPS on the
10,475 template vertices. You can run this right now, before you even run PEAR.

```bash
python3 -c "
from PEAR_PREPROCESSING import generate_global_files
generate_global_files(
    smplx_model_path='/path/to/smplx_models/',
    output_dir='/path/to/MVHumanNet++/',
    n_anchors=512
)
"
```

---

## Train / Test Split Strategy for 10 Sequences

With only 10 subject sequences, two complementary splits are used:
one to prove generalization to **unseen people** (primary),
one to prove generalization to **unseen poses** of seen people (secondary).

### Why Two Splits?

| Evaluation | What It Proves |
|---|---|
| **Primary** — held-out subjects | Model generalises to new body shapes/identities it has never seen |
| **Secondary** — held-out frames of same subjects | Model generalises to new poses of subjects it trained on |

Both are reported in the paper. Primary is the stronger claim.

---

### Split Assignment (10 Sequences)

```
Sequence IDs (example):  S01 S02 S03 S04 S05 S06 S07 S08 S09 S10

PRIMARY SPLIT (subject-disjoint):
  Train subjects : S01 S02 S03 S04 S05 S06 S07   (7 sequences, all frames)
  Val   subjects : S08                             (1 sequence,  all frames)
  Test  subjects : S09 S10                         (2 sequences, all frames)

SECONDARY SPLIT (same subjects, different frames):
  For EACH of the 7 train subjects:
    Train frames : first 80% of frames  (e.g. frames 0025–1200)
    Test  frames : last  20% of frames  (e.g. frames 1225–1500)
```

The secondary test set re-uses the same subjects as training but tests novel
poses they were never posed in during fine-tuning.

---

### Auto-Generate Split Files

Save this script as `scripts/generate_splits.py` and run it once:

```python
import os
import numpy as np

DATASET_ROOT = '/path/to/MVHumanNet++'
MAIN_ROOT    = os.path.join(DATASET_ROOT, 'main')
SPLITS_DIR   = os.path.join(DATASET_ROOT, 'splits')
os.makedirs(SPLITS_DIR, exist_ok=True)

# --- Discover all sequences and their frames ---
sequences = sorted([
    d for d in os.listdir(MAIN_ROOT)
    if os.path.isdir(os.path.join(MAIN_ROOT, d))
])
print(f"Found {len(sequences)} sequences: {sequences}")
assert len(sequences) == 10, "Expected exactly 10 sequences"

# --- Primary split: subject-disjoint ---
train_seqs = sequences[:7]   # S01–S07
val_seqs   = sequences[7:8]  # S08
test_seqs  = sequences[8:]   # S09–S10

with open(os.path.join(SPLITS_DIR, 'primary_train.txt'), 'w') as f:
    f.write('\n'.join(train_seqs))
with open(os.path.join(SPLITS_DIR, 'primary_val.txt'), 'w') as f:
    f.write('\n'.join(val_seqs))
with open(os.path.join(SPLITS_DIR, 'primary_test.txt'), 'w') as f:
    f.write('\n'.join(test_seqs))

print("Primary split written.")

# --- Secondary split: same subjects, held-out frames ---
# For each TRAIN subject, split frames 80/20
TRAIN_RATIO = 0.80
secondary_train_lines = []
secondary_test_lines  = []

for seq in train_seqs:
    cam00_dir  = os.path.join(MAIN_ROOT, seq, 'images', 'cam_00')
    all_frames = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(cam00_dir)
        if f.endswith('.jpg') and not f.startswith('A-')
    ])
    n_train = int(len(all_frames) * TRAIN_RATIO)
    for frame in all_frames[:n_train]:
        secondary_train_lines.append(f'{seq},{frame}')
    for frame in all_frames[n_train:]:
        secondary_test_lines.append(f'{seq},{frame}')

with open(os.path.join(SPLITS_DIR, 'secondary_train.txt'), 'w') as f:
    f.write('\n'.join(secondary_train_lines))
with open(os.path.join(SPLITS_DIR, 'secondary_test.txt'), 'w') as f:
    f.write('\n'.join(secondary_test_lines))

print(f"Secondary split written.")
print(f"  Train: {len(secondary_train_lines)} (seq, frame) pairs")
print(f"  Test:  {len(secondary_test_lines)} (seq, frame) pairs")

# --- Summary ---
print("\n=== SPLIT SUMMARY ===")
print(f"Primary   Train: {train_seqs}")
print(f"Primary   Val:   {val_seqs}")
print(f"Primary   Test:  {test_seqs}")
print(f"Secondary Train: {len(secondary_train_lines)} frame-pairs from {train_seqs}")
print(f"Secondary Test:  {len(secondary_test_lines)} frame-pairs from {train_seqs}")
```

Run:
```bash
python scripts/generate_splits.py
```

Output:
```
splits/
├── primary_train.txt     # 7 subject IDs, one per line
├── primary_val.txt       # 1 subject ID
├── primary_test.txt      # 2 subject IDs
├── secondary_train.txt   # "{seq},{frame}" pairs, 80% per train subject
└── secondary_test.txt    # "{seq},{frame}" pairs, 20% per train subject
```

---

### How the Dataloader Uses These Splits

```python
# PRIMARY training (subject-disjoint — pass sequence list):
dataset = get_anatomical_dataset(
    ...,
    sequences=open('splits/primary_train.txt').read().splitlines(),
)

# PRIMARY test (new humans — just change sequences):
test_dataset = get_anatomical_dataset(
    ...,
    sequences=open('splits/primary_test.txt').read().splitlines(),
)

# SECONDARY test (same humans, unseen frames):
# Load secondary_test.txt as explicit (seq, frame) pairs and filter
# in __getitem__ or build a separate filtered index.
```

---

### What to Report in the Paper

```
Table: Novel View Synthesis Quality

                    Primary (Unseen Subjects)    Secondary (Unseen Poses)
                    PSNR ↑  LPIPS ↓  Normal-L1 ↓   PSNR ↑  LPIPS ↓
Splatt3R (base)     xx.x    0.xxx    0.xxx           xx.x    0.xxx
+ Anatomical (ours) xx.x    0.xxx    0.xxx           xx.x    0.xxx
Δ improvement       +x.x    -0.xxx   -0.xxx          +x.x    -0.xxx
```

Primary improvement proves the GCN prior generalises across body identities.
Secondary improvement shows it also helps on new poses of familiar subjects.
