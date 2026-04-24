This design document outlines the implementation for the **Sharin-GAN** anatomical refinement module within the Splatt3R framework. The goal is to move from "pixel-only" reconstruction to "human-aware" reconstruction using SMPL-X priors.

---

### 1. System Architecture: Anatomical Conditioning
The architecture bridges the 3D graph representation of the human body (SMPL-X) with the 2D dense tokens of the MASt3R backbone.

#### A. The GCN Module (Anatomical Connectivity)
The GCN treats the human body as a graph where nodes are vertices and edges are physical bone/muscle connections.
* **Input:** 512 anchor vertices (subsampled via FPS from the 10,475 PEAR vertices) and the fixed SMPL-X adjacency matrix.
* **Process:** 2-3 layers of `GCNConv`. Each layer propagates features across the skeleton, ensuring the "elbow" node has context about the "wrist" and "shoulder."
* **Output:** A $512 \times 32$ feature matrix.



#### B. The PointNet++ Ablation (Spatial Density)
PointNet++ serves as the baseline to test if skeletal connectivity actually matters.
* **Input:** The same 512 anchor points, treated as an unordered point cloud.
* **Process:** Multi-scale grouping (MSG) and Set Abstraction layers. It captures local geometric density but has no concept of "joints" or "connectivity."
* **Ablation Goal:** If GCN > PointNet++, anatomical priors are confirmed as the superior "betterment" direction.



#### C. The Modified Gaussian Head
The `GaussianHead` in `catmlp_dpt_head.py` must be updated to accept the concatenated features.
* **Fusion:** Project the 512 3D features into the 2D image plane. Splat these into a $32 \times 32$ grid.
* **Concatenation:** New Input Dimension = $idim \text{ (1288)} + \text{Anatomical\_dim (32)} = 1320$.
* **Head:** Trainable DPT head that maps these 1320 channels to the standard Gaussian parameters ($\Delta, q, s, \alpha$).

---

### 2. Operational Modes

#### Train Mode (Finetuning)
* **Frozen:** MASt3R Encoder and Decoder (Activations offloaded via `torch.utils.checkpoint` to fit in 12GB).
* **Trainable:** GCN/PointNet++ encoder + Gaussian Prediction Head.
* **Batching:** Batch size 1-2 with Gradient Accumulation.
* **Data:** Triplets of (Context1, Context2, Target View).

#### Inference Mode (Zero-Shot)
* **Step 1:** Run PEAR on Input View 1 to generate the SMPL-X mesh.
* **Step 2:** Sample 512 anchors and run the GCN/PointNet++ encoder.
* **Step 3:** Feed MASt3R tokens and Projected Features into the head.
* **Step 4:** Render the Novel View.

---

### 3. Loss Modification
To avoid the "shrink-wrap" effect and preserve clothing/hair, we move away from standard SDF losses.

1.  **Masked Image Loss:**
    $$L_{img} = Mask_{human} \odot (\| \hat{I} - I \|^2 + \lambda L_{LPIPS})$$
    * *Source:* $Mask_{human}$ is derived from `.exr` depth files ($Depth > 0$), eroded by 3px.

2.  **Normal Consistency Loss:**
    $$L_{normal} = 1 - \cos(\hat{n}_{splat}, \hat{n}_{smplx})$$
    * *Source:* Compares the orientation of the rendered splat against the projected SMPL-X surface normals. This ensures the surface "faces" the right way without pulling Gaussians onto the skin surface.



---

### 4. Dataloader & Offline Preprocessing
The dataloader must handle the heterogeneous inputs from your `depth`, `main`, and `normal` folders.

#### Preprocessing Script (Run Once)
1.  **Mesh Extraction:** Run PEAR on every timestamp; save vertices as `.npz`.
2.  **Anchor Sampling:** Perform Farthest Point Sampling (FPS) on the 10k vertices down to 512. Store these indices.
3.  **Masking:** Threshold `.exr` files to create `.png` silhouettes.
4.  **Indexing:** Create a `dataset_index.json` containing valid (ViewA, ViewB, Target) triplets for every subject.

#### Dataloader Implementation (`__getitem__`)
* **Load Images:** RGB from `main/`.
* **Load Depth/Normals:** For target view supervision.
* **Load Geometry:** Load the 512 precomputed anchors and the fixed adjacency matrix.
* **Apply Transforms:** Random horizontal flips (adjusting camera extrinsics and mesh vertices accordingly) and brightness jitter for augmentation.

| Field | Source Folder | Use Case |
| :--- | :--- | :--- |
| `image` | `main/{id}` | Input to MASt3R |
| `depth` | `depth/{id}` | Mask generation & SDF (optional) |
| `normal` | `normal/{id}` | Normal Consistency Loss |
| `mesh` | `PEAR_output/` | GCN/PointNet++ input |

This implementation allows you to start training on the **ADA cluster** while staying strictly within the 12GB VRAM limit by leveraging the 512-anchor bottleneck and activation checkpointing.