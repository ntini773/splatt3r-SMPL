# Finetuning Plan: Direct Human-Bias + LoRA Decoder Adaptation

## Scope
This run path directly combines the old Step 1 and Step 2 into one implementation:
- Human token bias is injected before decoder blocks.
- LoRA is applied to early decoder self-attention.
- Loss masking uses only geom&human (with erosion kept).
- Boundary-ring weighting is removed.
- Background weighting uses two-zone soft weighting with 0.05 default and 0.1 ablation.
- Decoder input tokens are visualized in 2D using t-SNE during training.

## Implemented Changes
1. Decoder input conditioning:
- Added `human_type_embed` in `MAST3RAnatomicalRefinement`.
- Patchified `mask_human` to token labels and added learned token bias before decoder blocks.

2. Direct LoRA on decoder:
- Added fused-qkv LoRA wrapper (`LoRAQKVLinear`) to target q/v updates in self-attention.
- Applied LoRA to the first 2 blocks in both decoder branches (`dec_blocks` and `dec_blocks2`).

3. Masking simplification:
- Final mask remains `geom_mask & dataset_human_mask`.
- Erosion remains enabled.
- Removed boundary-ring logic from training loss.
- Added two-zone weighting helper with optional per-image normalization.

4. t-SNE diagnostics:
- Cached decoder input tokens after bias injection.
- Added periodic rank-0 t-SNE plot saving during visualization steps.

5. Config sync:
- Set default `background_weight: 0.05`.
- Added `lora` config block (blocks/rank/alpha/dropout).
- Added `viz` config block (`tsne_every_n_steps`, `tsne_min_step`, `tsne_max_tokens`).
- Removed boundary-ring semantics from config comments.

## Active Ablations
1. `background_weight=0.05` (primary)
2. `background_weight=0.1` (ablation)

## Verification Checklist
1. Smoke run starts without decoder-shape/mask-shape errors.
2. Trainable params include LoRA + human embedding and exclude frozen base decoder weights.
3. Visualizations include normal mask grids and periodic `tsne_step_*.png`.
4. Benchmarks use same two-zone weighting policy as training.

## Results Log
| Run | Trainable params | bg weight | Notes | Metrics |
|-----|------------------|-----------|-------|---------|
| Integrated LoRA + bias (primary) | TBD | 0.05 | direct combined run | TBD |
| Integrated LoRA + bias (ablation) | TBD | 0.1 | compare background influence | TBD |
