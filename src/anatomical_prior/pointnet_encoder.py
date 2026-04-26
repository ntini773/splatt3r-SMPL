"""
src/anatomical_prior/pointnet_encoder.py

PointNet++ segmentation backbone — drop-in replacement for AnatomicalGCNEncoder.

API contract (identical to GCN):
    encoder = AnatomicalPointNetEncoder(in_channels=3, hidden_channels=128, out_channels=32)
    out = encoder(x, adj=None)   # adj is silently ignored for API compatibility
    # x:   [B, N, 3]   mesh anchor 3-D coordinates (SMPL vertices, typically N=512)
    # out: [B, N, 32]  per-vertex feature vectors — same shape as GCN output

No torch_geometric / PyTorch3D / custom CUDA ops needed.
Fully compatible with Lightning DDP out-of-the-box.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest-Point Sampling.

    Args:
        xyz:    [B, N, 3]
        npoint: number of centroids to sample

    Returns:
        idx: [B, npoint]  integer indices into xyz
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), fill_value=1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)        # [B, 1, 3]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)            # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=-1).indices
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by index, supporting [B,S] and [B,S,K] idx tensors.

    Args:
        points: [B, N, C]
        idx:    [B, S] or [B, S, K]

    Returns:
        [B, S, C] or [B, S, K, C]
    """
    B = points.shape[0]
    device = points.device
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_idx, idx, :]


def ball_query(
    radius: float, nsample: int,
    xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Ball-query neighbourhood search.

    Args:
        radius:   search radius
        nsample:  max neighbours to keep
        xyz:      [B, N, 3]  all points
        new_xyz:  [B, S, 3]  query centroids

    Returns:
        group_idx: [B, S, nsample]  indices into xyz
                   Points not found within radius are replaced by the
                   nearest valid neighbour (index 0 of sorted list).
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    device = xyz.device

    dist = torch.cdist(new_xyz, xyz)                               # [B, S, N]
    base = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, N)
    masked = base.clone().float()
    masked[dist > radius] = float(N)                               # mark out-of-radius

    sorted_idx = masked.long().sort(dim=-1).values[:, :, :nsample] # [B, S, nsample]
    # Replace any remaining N-sentinels with the nearest in-radius index
    first = sorted_idx[:, :, :1].expand_as(sorted_idx)
    sorted_idx[sorted_idx >= N] = first[sorted_idx >= N]
    return sorted_idx


# ---------------------------------------------------------------------------
# Set Abstraction
# ---------------------------------------------------------------------------

def _sample_and_group(npoint, radius, nsample, xyz, points):
    """Returns new_xyz [B,npoint,3] and grouped [B,npoint,nsample,3+C]."""
    fps_idx = farthest_point_sample(xyz, npoint)                   # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)                           # [B, npoint, 3]
    idx = ball_query(radius, nsample, xyz, new_xyz)                # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)                           # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)          # relative coords

    if points is not None:
        grouped_pts = index_points(points, idx)                    # [B, npoint, nsample, C]
        new_points = torch.cat([grouped_xyz_norm, grouped_pts], dim=-1)
    else:
        new_points = grouped_xyz_norm                              # [B, npoint, nsample, 3]

    return new_xyz, new_points


def _sample_and_group_all(xyz, points):
    """Treat every point as one single group (global SA)."""
    B, N, _ = xyz.shape
    new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
    grouped_xyz = xyz.unsqueeze(1)                                 # [B, 1, N, 3]
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    One Set-Abstraction (SA) level of PointNet++.

    Performs: Sample → Group → PointNet MLP → MaxPool
    """

    def __init__(
        self,
        npoint,
        radius,
        nsample,
        in_channel: int,
        mlp_channels: list,
        group_all: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        layers = []
        prev = in_channel
        for out_ch in mlp_channels:
            layers += [
                nn.Conv2d(prev, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            prev = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, xyz, points):
        """
        xyz:    [B, N, 3]
        points: [B, N, C]  or  None

        Returns:
            new_xyz:    [B, npoint, 3]
            new_points: [B, npoint, out_channels]
        """
        if self.group_all:
            new_xyz, new_points = _sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = _sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_points: [B, npoint, nsample, C]
        new_points = new_points.permute(0, 3, 2, 1)               # [B, C, nsample, npoint]
        new_points = self.mlp(new_points)                          # [B, out_ch, nsample, npoint]
        new_points = new_points.max(dim=2).values                  # [B, out_ch, npoint]
        new_points = new_points.permute(0, 2, 1)                   # [B, npoint, out_ch]
        return new_xyz, new_points


# ---------------------------------------------------------------------------
# Feature Propagation
# ---------------------------------------------------------------------------

class PointNetFeaturePropagation(nn.Module):
    """
    Inverse-distance-weighted interpolation followed by a unit-pointnet MLP.
    Used to upsample from a coarser level back to a finer one.
    """

    def __init__(self, in_channel: int, mlp_channels: list):
        super().__init__()
        layers = []
        prev = in_channel
        for out_ch in mlp_channels:
            layers += [
                nn.Conv1d(prev, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            prev = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1:    [B, N, 3]   dense target positions
        xyz2:    [B, S, 3]   sparse source positions  (S < N)
        points1: [B, N, C1]  skip-connection features at target (may be None)
        points2: [B, S, C2]  features to interpolate from source

        Returns: [B, N, out_channels]
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # Global feature: simply broadcast
            interpolated = points2.expand(B, N, -1)
        else:
            # k=3 nearest neighbours, inverse-distance weights
            dist = torch.cdist(xyz1, xyz2)                         # [B, N, S]
            dist, idx = dist.topk(3, dim=-1, largest=False)        # [B, N, 3]
            dist = dist.clamp(min=1e-10)
            w = 1.0 / dist
            w = w / w.sum(dim=-1, keepdim=True)                    # [B, N, 3]
            nbr_feats = index_points(points2, idx)                 # [B, N, 3, C2]
            interpolated = (nbr_feats * w.unsqueeze(-1)).sum(dim=2) # [B, N, C2]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1)                   # [B, C, N]
        new_points = self.mlp(new_points)                          # [B, out_ch, N]
        return new_points.permute(0, 2, 1)                         # [B, N, out_ch]


# ---------------------------------------------------------------------------
# Top-level encoder  (drop-in for AnatomicalGCNEncoder)
# ---------------------------------------------------------------------------

class AnatomicalPointNetEncoder(nn.Module):
    """
    PointNet++ segmentation encoder for anatomical mesh priors.

    Architecture (N = 512 SMPL anchors by default):

        SA-1  : 512  → 128  (r=0.05 m ≈ 5 cm,  k=32 neighbours)
        SA-2  : 128  → 32   (r=0.20 m ≈ 20 cm, k=64 neighbours)
        SA-3  :  32  →  1   (global pooling)
        ──────────────────────────────────────────────────────────
        FP-3  :   1  → 32   (interpolate back to SA-2 resolution)
        FP-2  :  32  → 128  (interpolate back to SA-1 resolution)
        FP-1  : 128  → 512  (interpolate back to full N)
        ──────────────────────────────────────────────────────────
        out-mlp              per-point hidden → out_channels

    Parameters
    ----------
    in_channels      : 3 for bare XYZ; 6 if normals are appended
    hidden_channels  : base width (default 128)
    out_channels     : output feature dim per point (default 32, matches GCN)
    sa_npoints       : (n1, n2) downsample targets
    sa_radii         : (r1, r2) ball-query radii in the point-cloud's unit
    sa_nsamples      : (k1, k2) max neighbours per centroid
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        out_channels: int = 32,
        sa_npoints: tuple = (128, 32),
        sa_radii: tuple = (0.05, 0.20),
        sa_nsamples: tuple = (32, 64),
    ):
        super().__init__()
        H = hidden_channels
        n1, n2 = sa_npoints
        r1, r2 = sa_radii
        k1, k2 = sa_nsamples

        # ------------------------------------------------------------------
        # Downsampling path
        # ------------------------------------------------------------------
        # SA-1: N → n1   (relative-xyz only as group features at first layer)
        self.sa1 = PointNetSetAbstraction(
            npoint=n1, radius=r1, nsample=k1,
            in_channel=3,                          # just relative xyz; no extra feats at l0
            mlp_channels=[H // 2, H // 2, H],
        )
        # SA-2: n1 → n2
        self.sa2 = PointNetSetAbstraction(
            npoint=n2, radius=r2, nsample=k2,
            in_channel=3 + H,
            mlp_channels=[H, H, H * 2],
        )
        # SA-3: n2 → 1  (global)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=3 + H * 2,
            mlp_channels=[H * 2, H * 2, H * 4],
            group_all=True,
        )

        # ------------------------------------------------------------------
        # Upsampling path
        # ------------------------------------------------------------------
        # FP-3: 1 → n2
        self.fp3 = PointNetFeaturePropagation(
            in_channel=H * 2 + H * 4,             # sa2 skip + sa3 global
            mlp_channels=[H * 2, H * 2],
        )
        # FP-2: n2 → n1
        self.fp2 = PointNetFeaturePropagation(
            in_channel=H + H * 2,                 # sa1 skip + fp3 output
            mlp_channels=[H * 2, H],
        )
        # FP-1: n1 → N  (no skip because l0 has no learned features)
        self.fp1 = PointNetFeaturePropagation(
            in_channel=H,                          # fp2 output only (no skip at l0)
            mlp_channels=[H, H],
        )

        # ------------------------------------------------------------------
        # Per-point output head
        # ------------------------------------------------------------------
        self.out_mlp = nn.Sequential(
            nn.Conv1d(H, H, 1, bias=False),
            nn.BatchNorm1d(H),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Conv1d(H, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, adj=None) -> torch.Tensor:
        """
        Args:
            x:   [B, N, in_channels]   mesh anchor positions (and optionally normals)
            adj: ignored               kept for drop-in compatibility with GCN API

        Returns:
            [B, N, out_channels]
        """
        xyz = x[..., :3]                # positional coords
        l0_xyz = xyz
        l0_points = None                # no learned features at level-0 (raw XYZ only)

        # ---- Downsample ----
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)   # [B, n1, H]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)   # [B, n2, H*2]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)   # [B, 1,  H*4]

        # ---- Upsample ----
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, n2, H*2]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, n1, H]
        l0_points = self.fp1(l0_xyz, l1_xyz, None,       l1_points)  # [B, N,  H]

        # ---- Per-point head ----
        out = l0_points.permute(0, 2, 1)      # [B, H, N]
        out = self.out_mlp(out)               # [B, out_channels, N]
        return out.permute(0, 2, 1)           # [B, N, out_channels]