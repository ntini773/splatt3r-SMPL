import torch
from einops import rearrange, repeat

from .cuda_splatting import render_cuda, render_normals_cuda
from utils.geometry import normalize_intrinsics


class DecoderSplattingCUDA(torch.nn.Module):

    def __init__(self, background_color):
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, batch, pred1, pred2, image_shape):

        base_pose = batch['context'][0]['camera_pose']  # [b, 4, 4]
        inv_base_pose = torch.inverse(base_pose)

        extrinsics = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
        intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]

        # Rotate the ground truth extrinsics into the coordinate system used by MAST3R
        # --i.e. in the coordinate system of the first context view, normalized by the scene scale
        extrinsics = inv_base_pose[:, None, :, :] @ extrinsics

        # Merge Gaussians from both context views into a single cloud [B, G, ...]
        means       = torch.cat([rearrange(pred1["means"],              "b h w xyz -> b (h w) xyz"),
                                  rearrange(pred2["means_in_other_view"], "b h w xyz -> b (h w) xyz")], dim=1)
        covariances = torch.cat([rearrange(pred1["covariances"],  "b h w i j -> b (h w) i j"),
                                  rearrange(pred2["covariances"],  "b h w i j -> b (h w) i j")], dim=1)
        harmonics   = torch.cat([rearrange(pred1["sh"],  "b h w c d -> b (h w) c d"),
                                  rearrange(pred2["sh"],  "b h w c d -> b (h w) c d")], dim=1)
        opacities   = torch.cat([rearrange(pred1["opacities"],  "b h w 1 -> b (h w)"),
                                  rearrange(pred2["opacities"],  "b h w 1 -> b (h w)")], dim=1)

        b, v, _, _ = extrinsics.shape

        # ── Render target views ONE AT A TIME to cut peak VRAM by ~1/V ────────
        # The CUDA rasterizer allocates geomBuffer + binningBuffer + imgBuffer
        # per render call. Rendering v views together multiplies that footprint
        # by v. By looping we keep the allocator peak at 1× regardless of V.
        all_images = []
        for vi in range(v):
            torch.cuda.empty_cache()  # flush fragmented blocks before each rasterization
            ext_vi  = extrinsics[:, vi].contiguous()   # [b, 4, 4]
            intr_vi = intrinsics[:, vi].contiguous()   # [b, 3, 3]
            near_vi = torch.full((b,), 0.1,    device=means.device)
            far_vi  = torch.full((b,), 1000.0, device=means.device)

            img = render_cuda(
                ext_vi,
                intr_vi,
                near_vi,
                far_vi,
                image_shape,
                repeat(self.background_color, "c -> b c", b=b).contiguous(),
                means.contiguous(),
                covariances.contiguous(),
                harmonics.contiguous(),
                opacities.contiguous(),
            )  # [b, c, h, w]
            all_images.append(img)

        color = torch.stack(all_images, dim=1)  # [b, v, c, h, w]
        return color, None

    def render_normals(self, batch, pred1, pred2, image_shape):
        """
        Renders per-pixel surface normal maps for both context view Gaussians,
        projected into each target view's camera space.

        Normals are derived from each Gaussian's rotation quaternion (local Z axis)
        and alpha-composited using the same CUDA rasterizer via colors_precomp.
        This does NOT use SH evaluation — normals are passed as raw [G, 3] colors.

        Returns:
            normal_map: [B, V, 3, H, W] rendered normal maps, L2-normalised per pixel.
        """
        base_pose = batch['context'][0]['camera_pose']
        inv_base_pose = torch.inverse(base_pose)

        extrinsics = torch.stack([t['camera_pose'] for t in batch['target']], dim=1)
        intrinsics = torch.stack([t['camera_intrinsics'] for t in batch['target']], dim=1)
        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]
        extrinsics = inv_base_pose[:, None, :, :] @ extrinsics

        means       = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
        covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
        opacities   = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
        # rotations are [B, H, W, 4] quaternions; flatten to [B, G, 4]
        quats1 = rearrange(pred1["rotations"], "b h w q -> b (h w) q")
        quats2 = rearrange(pred2["rotations"], "b h w q -> b (h w) q")
        quaternions = torch.stack([quats1, quats2], dim=1)  # [B, 2, G, 4]

        b, v, _, _ = extrinsics.shape
        near = torch.full((b, v), 0.1, device=means.device)
        far  = torch.full((b, v), 1000.0, device=means.device)

        normal_map = render_normals_cuda(
            rearrange(extrinsics,   "b v i j -> (b v) i j").contiguous(),
            rearrange(intrinsics,   "b v i j -> (b v) i j").contiguous(),
            rearrange(near, "b v -> (b v)").contiguous(),
            rearrange(far,  "b v -> (b v)").contiguous(),
            image_shape,
            repeat(rearrange(means,       "b v h w xyz -> b (v h w) xyz"), "b g xyz -> (b v) g xyz", v=v).contiguous(),
            repeat(rearrange(covariances, "b v h w i j -> b (v h w) i j"), "b g i j -> (b v) g i j", v=v).contiguous(),
            repeat(rearrange(opacities,   "b v h w 1 -> b (v h w)"),       "b g -> (b v) g", v=v).contiguous(),
            repeat(rearrange(quaternions, "b v g q -> b (v g) q"),          "b g q -> (b v) g q", v=v).contiguous(),
        )
        normal_map = rearrange(normal_map, "(b v) c h w -> b v c h w", b=b, v=v)
        
        # Rotate normals from MASt3R's internal space (context[0] camera space)
        # to Absolute World Space to match the GT normals.
        R_c2w = base_pose[:, :3, :3]  # [B, 3, 3]
        normal_map = torch.einsum('b i j, b v j h w -> b v i h w', R_c2w, normal_map)
        
        return normal_map