# Splatt3R-GCN — Run Guide

> All commands are run from the project root: `/home2/gnrs/sfm/splatt3r_gcn`  
> Conda environment: `vision`  
> Main GPU node: `gnode043`

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Checkpoint Locations](#2-checkpoint-locations)
3. [Dataset Layout](#3-dataset-layout)
4. [2-View Inference (Base Splatt3R)](#4-2-view-inference-base-splatt3r)
5. [GCN Anatomical Refinement — Training](#5-gcn-anatomical-refinement--training)
6. [GCN Benchmark (2-view evaluation)](#6-gcn-benchmark-2-view-evaluation)
7. [N-View Mesh-Aware Inference](#7-n-view-mesh-aware-inference)
8. [Key Config Knobs](#8-key-config-knobs)
9. [Files to Change for Different Experiments](#9-files-to-change-for-different-experiments)
10. [Common Failure Modes & Fixes](#10-common-failure-modes--fixes)

---

## 1. Setup from Scratch (Clone & Run)

### Step 1 — Clone

```bash
# This repo lives on the 'splatt3r-gcn' branch of:
# https://github.com/ntini773/splatt3r-SMPL.git
git clone --branch splatt3r-gcn https://github.com/ntini773/splatt3r-SMPL.git splatt3r_gcn
cd splatt3r_gcn
```

### Step 2 — Environment

**Option A: Recreate exact environment (strongly recommended)**
```bash
conda env create -f environment_nview_gcn.yml
conda activate vision
```

> **⚠ Environment Compatibility Warning**  
> `environment_nview_gcn.yml` was exported from a Linux (x86_64) machine with **CUDA 12.1** and **Python 3.11**.  
> Key pinned versions: `torch==2.5.1+cu121`, `gsplat==1.5.3`, `lightning==2.6.1`, `scipy==1.14.1`  
> If your machine has a **different CUDA version**, the torch+cu121 packages will fail to install.  
> In that case use **Option B** and install the matching torch build for your CUDA.  
> Do **not** upgrade gsplat without testing — the rasterizer API changes between minor versions.

**Option B: Install from scratch (different CUDA version)**
```bash
conda env create -f environment.yml
conda activate vision
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

**Recompile CUDA RoPE kernels** (needed on first run):
```bash
cd src/mast3r_src/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../../
```

Set this before any training to prevent OOM fragmentation:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

### Step 3 — Download Checkpoints

```bash
bash scripts/download_checkpoints.sh
```

This downloads (to `checkpoints/`):

| File | Source | License | Access |
|---|---|---|---|
| `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth` | [NAVER MASt3R](https://github.com/naver/mast3r) | CC BY-NC-SA 4.0 | Public |
| `splatt3r_epoch19.ckpt` | [Splatt3R HuggingFace](https://huggingface.co/brandonsmart/splatt3r_v1.0) | CC BY-NC 4.0 | Public |
| `splatt3r_gcn_anatomical.ckpt` | [hf.co/ntini773/splatt3r-gcn](https://huggingface.co/ntini773/splatt3r-gcn) | CC BY-NC 4.0 | **Private — HF token required** |

> **SMPL-X model** requires manual registration at https://smpl-x.is.tue.mpg.de  
> Download v1.1 and place at `checkpoints/smplx/models/`

### Step 4 — Dataset

**MVHumanNet++ is a licensed dataset. Raw images cannot be redistributed.**

**Processed anchors** (SMPL-X vertex positions, no raw images) are hosted privately on HuggingFace at `ntini773/splatt3r-gcn-data`. Access is gated — only you can pull them with your token.

**Option A — Pull processed anchors (if you have access):**
```bash
huggingface-cli login   # paste your HF token (Settings → Access Tokens → write)
bash scripts/download_dataset.sh /your/output/path
```

**Option B — Preprocess from raw dataset yourself:**
```bash
# 1. Apply for access: https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet
# 2. Download to /your/path/mvhumannet++/
# 3. Run anchor preprocessing:
python third_party/PEAR/preprocess_mvhumannet_anchors.py \
  --dataset_root /your/path/mvhumannet++/ \
  --output_root /your/path/mvhumannet++_processed/ \
  --smplx_model_path checkpoints/smplx/models/
```

The files `smpl_adj_512.npy` and `fps_indices_512.npy` (SMPL-X topology, dataset-independent) are included in the repo under `data/` and require no download.

### Step 5 — Update Configs

Edit `configs/inference_nview_mesh_aware.yaml`:
```yaml
dataset_root: /your/path/mvhumannet++_processed
adj_path:     /your/path/mvhumannet++_processed/smpl_adj_512.npy
fps_indices_path: /your/path/mvhumannet++_processed/fps_indices_512.npy
smplx_model_path: checkpoints/smplx/models
model:        checkpoints/splatt3r_gcn_anatomical.ckpt
retrieval_ckpt: checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
output_dir:   /your/output/path
```

Edit `configs/benchmark_refined_splat3r.yaml`:
```yaml
splatt3r_checkpoint_path: checkpoints/splatt3r_epoch19.ckpt
refined_checkpoint_path:  checkpoints/splatt3r_gcn_anatomical.ckpt
```

### Step 6 — Test Run

```bash
python inference_nview_mesh_aware.py \
  --config configs/inference_nview_mesh_aware.yaml \
  --opt_depth True --opt_pose True --opt_focal True \
  --photometric_loss_w 0.1
```

Expected output: `scene.ply` and `scene_clean.ply` in `output_dir/<seq_id>/`

---

## ⚠ Hardcoded Paths Reference

> **If you are setting up on a new machine, update ALL paths below in the listed files.**  
> The configs currently point to paths on `gnode043` under `/ssd_scratch/gnrs/`. Change these to match your system.

| Config file | Key | Current hardcoded path | What to update to |
|---|---|---|---|
| `configs/inference_nview_mesh_aware.yaml` | `dataset_root` | `/ssd_scratch/gnrs/mvhumannet++_10demo` | Your MVHumanNet++ processed root |
| `configs/inference_nview_mesh_aware.yaml` | `adj_path` | `/ssd_scratch/gnrs/mvhumannet++_10demo/smpl_adj_512.npy` | Same root + `/smpl_adj_512.npy` |
| `configs/inference_nview_mesh_aware.yaml` | `fps_indices_path` | `/ssd_scratch/gnrs/mvhumannet++_10demo/fps_indices_512.npy` | Same root + `/fps_indices_512.npy` |
| `configs/inference_nview_mesh_aware.yaml` | `smplx_model_path` | `/ssd_scratch/gnrs/checkpoints/smplx/models` | Your SMPL-X model dir |
| `configs/inference_nview_mesh_aware.yaml` | `model` | `/ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=21-train/loss=0.3257.ckpt` | Downloaded GCN ckpt path |
| `configs/inference_nview_mesh_aware.yaml` | `output_dir` | `/ssd_scratch/gnrs/results_nview_gcn_test` | Your output directory |
| `configs/benchmark_refined_splat3r.yaml` | `splatt3r_checkpoint_path` | `/ssd_scratch/gnrs/checkpoints/epoch=19-step=1200.ckpt` | Downloaded Splatt3R ckpt path |
| `configs/benchmark_refined_splat3r.yaml` | `refined_checkpoint_path` | `/ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=21-train/loss=0.3257.ckpt` | Downloaded GCN ckpt path |
| `configs/benchmark_refined_splat3r.yaml` | `data.root` | `/ssd_scratch/gnrs/mvhumannet++_10demo` | Your dataset root |
| `configs/finetune_anatomical_mvhumannet.yaml` | `splatt3r_checkpoint_path` | `/ssd_scratch/gnrs/checkpoints/epoch=19-step=1200.ckpt` | Downloaded Splatt3R ckpt path |
| `configs/finetune_anatomical_mvhumannet.yaml` | `data.root` | `/ssd_scratch/gnrs/mvhumannet++_10demo` | Your dataset root |
| `train_anatomical_refinement.py` (L453) | checkpoint save dir | `/ssd_scratch/gnrs/gcn_checkpoints` | Your checkpoint save dir |

---

## How to Push Everything (First Time)

### 1 — Push code to GitHub

```bash
cd /home2/gnrs/sfm/splatt3r_gcn

git add -A
git commit -m "Add GCN anatomical refinement + N-view inference pipeline"

# Remote is already set (verify with: git remote -v)
# origin  https://github.com/ntini773/splatt3r-SMPL.git

# Push the splatt3r-gcn branch
git push -u origin splatt3r-gcn
```

### 2 — Login to HuggingFace

```bash
pip install huggingface_hub        # already in environment
huggingface-cli login              # paste your HF token (write access)
# Token location: https://huggingface.co/settings/tokens
```

### 3 — Upload checkpoint + dataset anchors (private repos)

```bash
bash scripts/upload_all_to_hf.sh
```

This creates two **private** repos (only accessible with your HF token):
- `ntini773/splatt3r-gcn`       → GCN checkpoint
- `ntini773/splatt3r-gcn-data`  → Processed anchor `.npy` files (no raw images)

---

## How to Re-Clone and Test from Scratch

```bash
# 1. Clone the splatt3r-gcn branch
git clone --branch splatt3r-gcn https://github.com/ntini773/splatt3r-SMPL.git splatt3r_gcn
cd splatt3r_gcn

# 2. Environment
conda env create -f environment_nview_gcn.yml
conda activate vision

# 3. HF login (needed for private checkpoint + dataset)
huggingface-cli login

# 4. Download all checkpoints (public + your private GCN ckpt)
bash scripts/download_checkpoints.sh

# 5. Download processed anchors (your private dataset repo)
bash scripts/download_dataset.sh /ssd_scratch/gnrs/mvhumannet++_10demo

# 6. Download raw MVHumanNet++ images (from official source, not hosted here)
#    https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet
#    Place at: /ssd_scratch/gnrs/mvhumannet++_10demo/main/<seq_id>/images/

# 7. SMPL-X (register at smpl-x.is.tue.mpg.de, place at checkpoints/smplx/models/)

# 8. Recompile CUDA kernels
cd src/mast3r_src/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../../

# 9. Test inference
python inference_nview_mesh_aware.py \
  --config configs/inference_nview_mesh_aware.yaml \
  --opt_depth True --opt_pose True --opt_focal True \
  --photometric_loss_w 0.1
```

---




## 2. Checkpoint Locations

| Checkpoint | Path |
|---|---|
| Base MASt3R weights | `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` |
| Retrieval (training-free) | `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth` |
| Base Splatt3R Lightning ckpt | `/ssd_scratch/gnrs/checkpoints/epoch=19-step=1200.ckpt` |
| GCN Refined ckpt (best) | `/ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=21-train/loss=0.3257.ckpt` |

> **Note:** When changing checkpoints, update paths in **both** the YAML config **and** verify the key `splatt3r_checkpoint_path` / `refined_checkpoint_path` / `model` match.

---

## 3. Dataset Layout

```
/ssd_scratch/gnrs/mvhumannet++_10demo/
├── main/
│   └── <seq_id>/          # e.g. 100831, 100835 ...
│       └── images/
│           └── cam_XX/    # cam_01, cam_11, cam_13, cam_14 ...
│               └── XXXX.jpg
├── depth/
│   └── <seq_id>/
├── anchors/
│   └── <seq_id>/          # pre-computed SMPL-X anchor .npy files
├── smpl_adj_512.npy        # fixed graph adjacency (512 FPS vertices)
└── fps_indices_512.npy     # FPS vertex indices into full SMPL-X mesh
```

Available sequence IDs: `100831` – `100840`

---

## 4. 2-View Inference (Base Splatt3R)

Runs the original unmodified Splatt3R on a pair of images.

```bash
python demo_cli.py \
  --image1 path/to/image1.jpg \
  --image2 path/to/image2.jpg \
  --output output_dir/
```

Or interactive Gradio demo:
```bash
python demo.py
```

---

## 5. GCN Anatomical Refinement — Training

Fine-tunes the GCN cross-attention + Gaussian DPT heads on the MVHumanNet++ dataset.

### Standard run (30 epochs, 2 GPUs)

```bash
python train_anatomical_refinement.py \
  configs/finetune_anatomical_mvhumannet.yaml
```

### Single-GPU debug run

```bash
python train_anatomical_refinement.py \
  configs/finetune_anatomical_mvhumannet.yaml \
  devices=[0]
```

### Resume from checkpoint

```bash
python train_anatomical_refinement.py \
  configs/finetune_anatomical_mvhumannet.yaml \
  ckpt_path=/ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=XX-train/loss=X.XXXX.ckpt
```

### Key training config (`configs/finetune_anatomical_mvhumannet.yaml`)

```yaml
splatt3r_checkpoint_path: '/ssd_scratch/gnrs/checkpoints/epoch=19-step=1200.ckpt'
data:
  root: '/ssd_scratch/gnrs/mvhumannet++_10demo'
  sequences: ['100831', '100832', ...]   # null = all sequences
  batch_size: 1
  num_target_views: 2
  epochs_per_train_epoch: 10
opt:
  epochs: 30
  lr: 0.00002
  accumulate_grad_batches: 8
loss:
  mse_loss_weight: 1.0
  lpips_loss_weight: 0.5
  background_weight: 0.1          # decays linearly to bg_weight_floor
  bg_weight_floor: 0.05
  boundary_weight: 0.3
  bg_schedule_end_epoch: 20
  bg_schedule_psnr_gate: 7.0
  apply_mask: True
  average_over_mask: True
```

**What is trainable:**
| Module | Status | LR |
|---|---|---|
| ViT backbone + cross-attn decoder | ❄️ Frozen | — |
| GCN encoder (`gcn_encoder`) | ✅ Trained from scratch | `lr` |
| `prior_attention` (both heads) | ✅ Trained from scratch | `lr` |
| Gaussian DPT (`gaussian_dpt.dpt`) | ✅ Fine-tuned | `lr × 0.1` |

---

## 6. GCN Benchmark (2-view evaluation)

Evaluates the refined GCN model on held-out sequences with PSNR / SSIM / LPIPS.

### Run refined benchmark

```bash
python refined_splatt3r_benchmark.py \
  configs/benchmark_refined_splat3r.yaml
```

### Run base Splatt3R benchmark (for comparison)

```bash
python base_splatt3r_benchmark.py \
  configs/benchmark_base_splat3r.yaml
```

### Key benchmark config (`configs/benchmark_refined_splat3r.yaml`)

```yaml
splatt3r_checkpoint_path: '/ssd_scratch/gnrs/checkpoints/epoch=19-step=1200.ckpt'
refined_checkpoint_path: '/ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=21-train/loss=0.3257.ckpt'
eval_save_dir: '/ssd_scratch/gnrs/refined_splatt3r_eval_results_second_test'
data:
  sequences: ['100840']    # sequences to evaluate on
  batch_size: 2
```

> Results saved to `eval_save_dir/`: `results.json`, `metrics.txt`, `visualizations/*.jpg`

---

## 7. N-View Mesh-Aware Inference

The main N-view pipeline: runs GCN Splatt3R on all pairwise view combinations, then
jointly optimises camera poses, depths, and Gaussian attributes via photometric + 3D
matching losses to produce a single world-space scene.

### Quick run (defaults from YAML)

```bash
python inference_nview_mesh_aware.py \
  --config configs/inference_nview_mesh_aware.yaml
```

### With geometry optimisation enabled (recommended for cleaner results)

```bash
python inference_nview_mesh_aware.py \
  --config configs/inference_nview_mesh_aware.yaml \
  --opt_depth True \
  --opt_pose True \
  --opt_focal True \
  --photometric_loss_w 0.1
```

### Deterministic single sequence

```bash
python inference_nview_mesh_aware.py \
  --config configs/inference_nview_mesh_aware.yaml \
  --sequence_id 100835 \
  --view_ids cam_13 cam_14 cam_11 cam_01 \
  --num_views 4
```

### Run all sequences (batch)

```bash
python inference_nview_mesh_aware.py \
  --config configs/inference_nview_mesh_aware.yaml \
  --random_batch False \
  --max_batches 10      # number of sequences to process
```

### Full command reference

```
--config                  Path to YAML config file
--dataset_root            Override dataset root path
--sequence_id             Fix a specific sequence (e.g. 100835)
--view_ids                Fix specific cameras (e.g. cam_13 cam_14 cam_11 cam_01)
--num_views               Number of views (default: 4)
--photometric_loss_w      Weight for photometric loss during optimisation (0 = off)
--opt_depth               True/False — optimise depth maps in fine pass
--opt_pose                True/False — optimise camera extrinsics
--opt_focal               True/False — optimise focal lengths
--random_batch            True/False — random vs sequential sequence selection
--max_batches             Max scenes to process
--output_dir              Override output directory
--model                   Path to GCN refined checkpoint
--retrieval_ckpt          Path to retrieval checkpoint (null = complete graph pairing)
```

### Key N-view config (`configs/inference_nview_mesh_aware.yaml`)

```yaml
include: ['benchmark_refined_splat3r.yaml']

dataset_root: /ssd_scratch/gnrs/mvhumannet++_10demo
adj_path: /ssd_scratch/gnrs/mvhumannet++_10demo/smpl_adj_512.npy
fps_indices_path: /ssd_scratch/gnrs/mvhumannet++_10demo/fps_indices_512.npy
smplx_model_path: /ssd_scratch/gnrs/checkpoints/smplx/models

model: /ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=21-train/loss=0.3257.ckpt
retrieval_ckpt: checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth

num_views: 4
photometric_loss_w: 0.0    # 0 = off, 0.1 = light, 0.5 = aggressive
opt_depth: false           # true = fix human depth inconsistency
opt_pose: false            # true = refine camera extrinsics
opt_focal: false           # true = refine focal lengths
random_batch: true
max_batches: 1
resolution: 512
output_dir: /ssd_scratch/gnrs/results_nview_gcn_test
```

### N-view outputs (per sequence, under `output_dir/<seq_id>/`)

```
scene.ply              # full Gaussian point cloud (confidence-filtered)
scene_clean.ply        # quality-filtered: conf + positive depth + opacity > 0.05
ply_export_stats.json  # per-view: n_total, n_conf_kept, n_neg_depth, n_low_opacity
view_000_rendered.png  # side-by-side: GT | Rendered | Depth (per view)
inference_summary.json # avg PSNR, avg LPIPS, metadata
```

> Open `scene_clean.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor) or [this viewer](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php) for 3DGS visualisation.

---

## 8. Key Config Knobs

### Optimisation stability (sparse_ga.py — auto, no change needed)

| Behaviour | Controlled by |
|---|---|
| Depth collapse prevention | Scale regulariser: `loss += 1e-3 * (sizes-1)².mean()` |
| Negative depth / streaks | Depth positivity penalty: `loss += 0.1 * relu(0.05 - depth).mean()` |
| Degenerate Gaussians | Clamping after step: `scales ∈ [1e-4, 0.5]`, `opacities ∈ [-5, 5]` |
| Noisy quaternion aggregation | Best-confidence selection (not weighted average) |

### What to change for cleaner N-view geometry

```yaml
# inference_nview_mesh_aware.yaml
opt_depth: true          # ← single most impactful change
opt_pose: true
opt_focal: true
photometric_loss_w: 0.1  # start low; 0.5 is too aggressive with wrong depths
```

### What to change for training on a new dataset

```yaml
# finetune_anatomical_mvhumannet.yaml
data:
  root: '/path/to/new/dataset'
  sequences: null          # auto-discover all sequences
  batch_size: 1
smplx_model_path: '/path/to/smplx/models'
adj_path: '/path/to/smpl_adj_512.npy'
fps_indices_path: '/path/to/fps_indices_512.npy'
```

---

## 9. Files to Change for Different Experiments

| Goal | File(s) to edit |
|---|---|
| Change model checkpoint | `configs/inference_nview_mesh_aware.yaml` → `model:` |
| Change base Splatt3R ckpt | `configs/benchmark_refined_splat3r.yaml` → `splatt3r_checkpoint_path:` |
| Change output directory | `configs/inference_nview_mesh_aware.yaml` → `output_dir:` |
| Change sequences evaluated | `configs/benchmark_refined_splat3r.yaml` → `data.sequences:` |
| Tune loss weights (training) | `configs/finetune_anatomical_mvhumannet.yaml` → `loss.*` |
| Tune background schedule | `configs/finetune_anatomical_mvhumannet.yaml` → `loss.bg_*` |
| Add/remove trainable modules | `train_anatomical_refinement.py` → `__init__` frozen/unfrozen sections |
| Change optimisation passes (N-view) | `src/mast3r_src/mast3r/cloud_opt/sparse_ga.py` → `sparse_scene_optimizer` args |
| Change depth regulariser strength | `sparse_ga.py` → `optimize_loop` → scale regulariser weight (`1e-3`) |
| Change depth positivity floor | `sparse_ga.py` → `optimize_loop` → `_depth_floor = 0.05` |
| Change PLY confidence threshold | Call site in `inference_nview_mesh_aware.py` → `export_scene_ply(..., conf_thresh=1.5)` |
| Change Gaussian scale clamp | `train_anatomical_refinement.py` → `pred1['scales'] = torch.clamp(...)` |
| Change GCN architecture | `src/anatomical_prior/gcn_encoder.py` |
| Change cross-attention injection | `src/anatomical_prior/gaussian_head.py` (or equivalent head file) |

---

## 10. Common Failure Modes & Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA illegal memory access` during training | Non-contiguous tensor passed to CUDA rasterizer | Add `.contiguous()` before rasterization call |
| OOM during training | Gaussian blowup from unconstrained scales | `scales = torch.clamp(scales, max=0.05)` in forward |
| N-view PLY is a flat slab | `opt_depth=False` → depth maps never reconcile across views | Set `opt_depth: true` in YAML |
| White streak artefacts in PLY | Negative depth values → points projected behind cameras | Depth positivity penalty (already in `sparse_ga.py`); use `scene_clean.ply` |
| Human pushed to background | Background (80% of frame) dominates 3D matching loss | Enable `opt_depth`, lower `photometric_loss_w` |
| Low SSIM with masking | Binary mask creates artificial edges in SSIM patch windows | Expected behaviour — PSNR metric is more meaningful here |
| `conf_mean` very low (< 1.5) for some views | Poor view overlap in the pair for that camera | Use `retrieval_ckpt` for smarter pair selection instead of complete graph |
| `means grad is None` in logs | `core_depth` not in optimiser params (opt_depth=False) | Expected when `opt_depth=False`; enable it to see depth gradients |
| `FutureWarning: weights_only=False` | PyTorch version mismatch | Ignorable warning; add `weights_only=True` to `torch.load` calls if needed |
