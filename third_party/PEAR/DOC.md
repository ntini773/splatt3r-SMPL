# MVHumanNet++ → Splatt3R (Human-Centric) Design Doc

## 0. Goal

Convert MVHumanNet++ into a **human-aware Splatt3R training pipeline** with:

* Reliable **human-only supervision**
* **PEAR-based geometry priors (anchors + normals)**
* Efficient **triplet sampling (context → target)**
* Compatibility with existing Splatt3R dataloader abstractions
* Fit within **12 GB VRAM**

---

## 1. Raw Dataset Understanding

Given structure:

```
depth/{seq}/{cam}/frame.exr
main/{seq}/images/{cam}/frame.jpg
normal/{seq}/{cam}/frame.png or exr
```

Each sequence:

* 16 synchronized cameras
* Same frame across all cameras
* Human centered in scene

---

## 2. Target Processed Dataset Structure

We create a **new processed root** (do NOT modify original):

```
processed_root/
  {seq}/
    {frame}/
      cam_00.npz
      cam_01.npz
      ...
```

Each `.npz` contains:

```
{
  image: (H,W,3) uint8
  depth: (H,W) float32
  K: (4,4)
  c2w: (4,4)

  mask_human: (H,W) uint8
  normals: (H,W,3) float32

  anchors: (512,3) float32
}
```

---

## 3. Offline Processing Pipeline

### 3.1 Human Mask

From depth:

```
mask = (depth > 0)
```

Refinement:

* erosion (3x3)
* optional dilation (1–2 px)

Output:

```
mask_human ∈ {0,1}^{H×W}
```

---

### 3.2 PEAR Mesh Extraction

For each **(seq, frame)**:

1. Run PEAR on multi-view inputs
2. Get mesh (~10k vertices)

---

### 3.3 Anchor Generation (FPS)

Apply Farthest Point Sampling:

```
10475 → 512 anchors
```

Store:

```
anchors: (512,3)
```

NOTE:

* Same anchors reused for all 16 cameras of same frame

---

### 3.4 Normal Map

Render PEAR mesh into each camera:

```
normals: (H,W,3)
```

---

### 3.5 Why per-camera .npz?

* Avoid runtime disk joins
* Faster IO
* Simpler dataloader

---

## 4. View Sampling Strategy (CRITICAL)

We MUST replace random camera sampling.

### Camera geometry

* 16 cams → 360°
* Δθ = 22.5°

---

### 4.1 Training Triplets

#### Stage 1 (default)

```
Context: (i, i+1)
Target:  (i+2)
```

#### Stage 2

```
Context: (i, i+2)
Target:  (i+4)
```

---

### 4.2 Sampling Implementation

```
i = randint(0,15)
k ∈ {1,2}

c1 = i
c2 = (i+k)%16
t  = (i+2*k)%16
```

---

### 4.3 Why this works

* Humans self-occlude heavily
* Small angle → dense valid mask
* Large angle → mask collapses

---

## 5. New Dataloader Design

We DO NOT reuse old dataloader.

### 5.1 Output format (important)

Return EXACT Splatt3R-compatible structure:

```
{
  context: [view1, view2],
  target:  [view3],
  anchors: (512,3),
  scene: seq
}
```

---

### 5.2 Context view fields

```
{
  img
  original_img
  depthmap
  pts3d
  valid_mask

  camera_pose
  camera_intrinsics

  mask_human
  normals
}
```

---

### 5.3 Target view fields

```
{
  original_img
  camera_pose
  camera_intrinsics

  mask_human
  normals
}
```

---

## 6. Mask Integration (CORE)

Original:

```
M_total = frustum ∧ depth ∧ valid
```

New:

```
M_final = M_total ∧ mask_human
```

---

### Practical version (recommended)

```
M_final = M_total * (0.7 + 0.3 * mask_human)
```

---

## 7. Loss Functions

### 7.1 RGB Loss

```
L_rgb = mask * |render - gt|
```

---

### 7.2 Normal Loss

```
L_normal = mask_human * |N_render - N_smpl|
```

---

### 7.3 Anchor Loss

```
L_anchor = min distance (gaussians ↔ anchors)
```

---

### Final

```
L = L_rgb + λ1 L_normal + λ2 L_anchor
```

---

## 8. Model Changes

### Minimal (recommended)

Inject anchors as global feature:

```
anchor_feat = MLP(mean(anchors))
```

Concatenate with encoder output

---

### Optional (better)

Cross-attention:

* image tokens ↔ anchors (512)
* keep ≤ 2 layers

---

## 9. VRAM Strategy

* anchors small (512×3)
* fp16 mandatory
* avoid large attention blocks
* reuse anchors across views

---

## 10. Training Flow

```
sample (seq, frame)
  ↓
sample (c1,c2,t)
  ↓
load npz
  ↓
predict Gaussians
  ↓
render target
  ↓
compute mask
  ↓
compute loss
  ↓
backprop
```

---

## 11. Key Failure Modes

| Issue        | Cause           | Fix          |
| ------------ | --------------- | ------------ |
| No gradients | mask too strict | soften mask  |
| Floaters     | no normals      | add L_normal |
| Collapse     | large view gap  | fix sampling |

---

## 12. Final Insight

This pipeline turns Splatt3R into:

> "Geometry-aware human reconstruction system with masked supervision and anatomical priors"

The **most critical pieces** are:

1. View sampling (small angular gaps)
2. Mask quality (not too strict)
3. Lightweight anchor usage

Everything else is secondary.
