"""
geometry/mask_gaussians.py
--------------------------
Filters Gaussian parameter dicts to retain only foreground (human) pixels.

CONTRACT
--------
Input:
    gaussian_params : dict[str, Tensor(B, H, W, ...)]
        Raw prediction dict from _downstream_head. Expected keys:
            'means'       (B, H, W, 3)
            'scales'      (B, H, W, 3)
            'rotations'   (B, H, W, 4)
            'sh'          (B, H, W, 3, D_sh)
            'opacities'   (B, H, W, 1)
            'covariances' (B, H, W, 3, 3)
        Optional passthrough keys (e.g. 'pts3d', 'pts3d_in_other_view',
        'conf', 'means_in_other_view') are forwarded as-is.

    foreground_mask : Tensor(B, H, W) bool
        True  = human / keep
        False = background / discard
        Must already be on the same device as gaussian_params tensors.

Output:
    List[Dict[str, Tensor]]   length = B
        Per-batch-item dicts with only the foreground Gaussians:
            'means'       (N_i, 3)
            'scales'      (N_i, 3)
            'rotations'   (N_i, 4)
            'sh'          (N_i, 3, D_sh)
            'opacities'   (N_i, 1)
            'covariances' (N_i, 3, 3)

DESIGN NOTES
------------
- Hard removal (not zero-opacity masking). Zero-opacity Gaussians still
  participate in sorting, memory allocation, and gradient computation.
- Covariances are kept as (N_i, 3, 3) structured matrices — the renderer
  indexes them via torch.triu_indices(3, 3) to extract the upper triangle.
  Do NOT pre-flatten to (N_i, 6) here.
- Optional dilation (kernel_size=3) can be enabled to preserve boundary
  pixels where masks are noisy at silhouette edges.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# Keys that have spatial layout (B, H, W, ...) and should be filtered.
_SPATIAL_KEYS = {
    'means', 'scales', 'rotations', 'sh', 'opacities', 'covariances',
    'means_in_other_view',   # pred2's positional tensor: (B,H,W,3), must be spatially filtered
}

# Keys that are already in a different format (flat / non-spatial) and
# should be passed through without modification.
_PASSTHROUGH_KEYS = {
    'pts3d', 'pts3d_in_other_view',
    'conf', 'conf2',
}


def dilate_mask(mask: Tensor, kernel_size: int = 3) -> Tensor:
    """
    Morphological dilation of a boolean mask using max-pooling.

    Args:
        mask       : (B, H, W) bool tensor
        kernel_size: dilation radius (odd integer)

    Returns:
        dilated mask (B, H, W) bool
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    padding = kernel_size // 2
    # max-pool on float, then re-binarise
    dilated = F.max_pool2d(
        mask.float().unsqueeze(1),   # (B, 1, H, W)
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    ).squeeze(1)                      # (B, H, W)
    return dilated > 0.5


def mask_gaussians(
    gaussian_params: dict[str, Tensor],
    foreground_mask: Tensor,
    dilate: bool = False,
    dilate_kernel: int = 3,
) -> list[dict[str, Tensor]]:
    """
    Hard-filter Gaussian parameters to foreground pixels only.

    Parameters
    ----------
    gaussian_params   : dict of (B, H, W, ...) tensors from _downstream_head.
    foreground_mask   : (B, H, W) bool tensor, already on the correct device.
                        True = human pixel to keep.
    dilate            : If True, dilate the mask slightly before filtering
                        to preserve boundary pixels at silhouetter edges.
    dilate_kernel     : Dilation kernel size (must be odd). Default 3 → 1px border.

    Returns
    -------
    List of length B.  Each element is a dict[str, Tensor] containing only
    the foreground Gaussians for that batch item:
        means       (N_i, 3)
        scales      (N_i, 3)
        rotations   (N_i, 4)
        sh          (N_i, 3, D_sh)
        opacities   (N_i, 1)
        covariances (N_i, 3, 3)
    """
    # ── Validate mask ────────────────────────────────────────────────────────
    assert foreground_mask.dtype == torch.bool, (
        f"foreground_mask must be bool, got {foreground_mask.dtype}. "
        "Use: torch.as_tensor(mask_human, device=device).bool()"
    )

    B, H, W = foreground_mask.shape

    if dilate:
        foreground_mask = dilate_mask(foreground_mask, kernel_size=dilate_kernel)

    # Flatten spatial dims for index selection: (B, H*W)
    mask_flat = foreground_mask.reshape(B, -1)

    result: list[dict[str, Tensor]] = []

    for b in range(B):
        m = mask_flat[b]       # (H*W,) bool — foreground selector for item b
        fg_count = m.sum().item()
        item: dict[str, Tensor] = {}

        for key, val in gaussian_params.items():
            if key in _PASSTHROUGH_KEYS:
                # Non-spatial tensors: pass through as-is (not per-pixel)
                item[key] = val[b] if val.dim() > 0 and val.shape[0] == B else val
                continue

            if key not in _SPATIAL_KEYS:
                # Unknown key — pass through without filtering to avoid silent drops
                item[key] = val[b] if val.dim() > 0 and val.shape[0] == B else val
                continue

            # ── Validate spatial shape ────────────────────────────────────
            assert val.shape[:3] == (B, H, W), (
                f"Key '{key}': expected shape ({B}, {H}, {W}, ...), "
                f"got {tuple(val.shape)}"
            )

            # Flatten spatial dims: (B, H*W, ...)
            extra = val.shape[3:]           # trailing dims, e.g. (3,) or (3,3) or (3,D)
            val_flat = val.reshape(B, H * W, *extra)

            # Select foreground pixels for this batch item
            item[key] = val_flat[b][m]      # (N_i, *extra)

        result.append(item)

    return result


# ---------------------------------------------------------------------------
# Padding utility — used by the decoder to create a dense batch from ragged lists
# ---------------------------------------------------------------------------

def pad_masked_to_dense(
    masked_list: list[dict[str, Tensor]],
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    """
    Pad a ragged list of per-item Gaussian dicts to a dense batch tensor.

    All padding entries have:
        opacities  = 0.0   ← critical: makes them invisible to the renderer
        means      = 0.0
        scales     = 1e-6  ← tiny but non-zero (avoids degenerate covariance)
        everything else = 0.0

    Parameters
    ----------
    masked_list : List[Dict[str, Tensor]]   length B, from mask_gaussians()
    device      : target device for output tensors

    Returns
    -------
    padded : Dict[str, Tensor]   shape (B, N_max, ...)
    counts : Tensor(B,) int      number of real Gaussians per item
    """
    B = len(masked_list)
    counts = torch.tensor([item['means'].shape[0] for item in masked_list],
                          dtype=torch.long, device=device)
    N_max = int(counts.max().item())

    if N_max == 0:
        # Edge case: no foreground pixels at all (should not happen with valid masks)
        # Return a single dummy Gaussian per item with zero opacity
        N_max = 1

    # Allocate output tensors (zeros by default)
    # We'll fill real entries first, padding remains zero.
    keys = list(masked_list[0].keys())
    padded: dict[str, Tensor] = {}

    for key in keys:
        if key in _PASSTHROUGH_KEYS:
            continue
        ref = masked_list[0][key]
        extra = ref.shape[1:]               # trailing dims after N_i
        padded[key] = torch.zeros(B, N_max, *extra, dtype=ref.dtype, device=device)

    # Fill in per-item foreground data
    for b, item in enumerate(masked_list):
        N_i = item['means'].shape[0]
        if N_i == 0:
            continue
        for key in keys:
            if key in _PASSTHROUGH_KEYS:
                continue
            padded[key][b, :N_i] = item[key].to(device)

    # ── Enforce zero opacity on ALL padding entries ───────────────────────
    # This is the critical safety check: padding slots must be invisible.
    for b in range(B):
        N_i = counts[b].item()
        if N_i < N_max:
            padded['opacities'][b, N_i:] = 0.0

    # Set scales to tiny non-zero for padding (avoids singular covariance matrix)
    for b in range(B):
        N_i = counts[b].item()
        if N_i < N_max and 'scales' in padded:
            padded['scales'][b, N_i:] = 1e-6

    return padded, counts
