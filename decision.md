# Decision: Two-Zone Probability-Mix Weighting vs True Exp-Softmax

## Context
We replaced boundary-ring masking with a two-zone mask policy:
- Foreground zone: eroded intersection mask `(geom_mask ∩ human_mask)`
- Background zone: complement of foreground

We needed a soft weighting policy using background weight candidates `0.05` and `0.1`.

## Chosen Method
Use probability-mix style weighting:

- `soft_mask = human_mask + bg_weight * (1 - human_mask)`
- `bg_weight` is typically `0.05` (primary) or `0.1` (ablation)
- Loss uses weighted averaging over this soft mask

This method keeps the weight meaning direct and stable: foreground is always weight `1.0`, background is exactly the configured scalar.

## Why This Is Better Here
1. Interpretability: `bg_weight=0.05` literally means background contributes `5%` of human-region weight per pixel.
2. Stable optimization: no exponential transforms, so gradients do not become unexpectedly sharp/flat due to logit scaling.
3. Reproducibility across runs: behavior is less sensitive to dataset-dependent logit distributions.
4. Easy ablation: moving from `0.05` to `0.1` is a clear linear change, making comparisons cleaner.

## Why Not True Exp-Softmax As Default
A strict exp-softmax setup requires logits and temperature choices. In practice:
1. Calibration ambiguity: `0.05` and `0.1` do not map linearly to final probabilities unless logit mapping is explicitly engineered.
2. Exponential sensitivity: small logit changes can strongly alter effective weighting.
3. Harder cross-run comparability: probability mass depends on both class logits and spatial composition, not just the intended scalar background weight.
4. Unnecessary complexity: no clear evidence it improves this objective versus the simpler weighted-average mask.

## Optional Future Work
If desired, we can add an experimental mode with true exp-softmax weighting behind a config flag and compare against the chosen default on the same checkpoints.
