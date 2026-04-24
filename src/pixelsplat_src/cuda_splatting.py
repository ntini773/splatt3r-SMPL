from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from torch import Tensor

from .projection import get_fov, homogenize_points


def get_projection_matrix(
    near,
    far,
    fov_x,
    fov_y,
):
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussian_sh_coefficients,
    gaussian_opacities,
    scale_invariant: bool = True,
    use_sh: bool = True,
):
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


def render_cuda_orthographic(
    extrinsics,
    width,
    height,
    near,
    far,
    image_shape: tuple[int, int],
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussian_sh_coefficients,
    gaussian_opacities,
    fov_degrees,
    use_sh: bool = True,
    dump: dict | None = None,
):
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def quaternion_to_surface_normal(quaternions: Tensor) -> Tensor:
    """
    Extracts the surface normal (shortest axis / local Z) from each Gaussian's
    rotation quaternion using the standard quaternion-to-rotation formula.

    WHY THIS IS THE NORMAL
    ----------------------
    A 3D Gaussian is an ellipsoid. Its "surface" at any point faces along the
    axis that has the smallest scale (the most disc-like direction). By convention
    in Gaussian Splatting, this is the local Z axis after applying the quaternion
    rotation. The third column of the rotation matrix gives this direction in
    world space.

    Args:
        quaternions: [..., 4] tensor in (qw, qx, qy, qz) order.

    Returns:
        normals: [..., 3] unit vectors in world space.
    """
    qw, qx, qy, qz = quaternions.unbind(-1)
    # Third column of rotation matrix from quaternion:
    nx = 2 * (qx * qz + qw * qy)
    ny = 2 * (qy * qz - qw * qx)
    nz = 1 - 2 * (qx ** 2 + qy ** 2)
    return torch.stack([nx, ny, nz], dim=-1)  # [..., 3], already unit length


def render_normals_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    gaussian_means,
    gaussian_covariances,
    gaussian_opacities,
    gaussian_quaternions,
    scale_invariant: bool = True,
) -> Tensor:
    """
    Renders a per-pixel surface normal map by alpha-compositing per-Gaussian
    normals using the same CUDA rasterizer as RGB rendering.

    HOW IT WORKS
    ------------
    The `colors_precomp` argument of GaussianRasterizer accepts a raw [G, 3]
    tensor that bypasses SH evaluation entirely. Since normals are already
    3-vectors (nx, ny, nz), we pass them directly as `colors_precomp`.
    The kernel alpha-composites them identically to RGB — the output pixel
    value is the opacity-weighted average normal of all Gaussians covering
    that pixel, which is the standard definition of a rendered normal map.

    SH colors cannot be used here because the SH evaluation shader treats
    the coefficients as view-dependent color harmonics — meaningless for normals.

    Args:
        gaussian_quaternions: [B, G, 4] rotation quaternions (qw, qx, qy, qz)

    Returns:
        normal_map: [B, 3, H, W], values in [-1, 1], L2-normalised per pixel.
    """
    # Derive per-Gaussian surface normals from their rotation quaternions.
    # Shape: [B, G, 3]
    normals_per_gaussian = quaternion_to_surface_normal(gaussian_quaternions)

    # Fake SH: wrap the [B, G, 3] normals into the [B, G, 3, 1] shape that
    # render_cuda expects for `gaussian_sh_coefficients` (degree-0 = constant color).
    fake_sh = normals_per_gaussian.unsqueeze(-1)  # [B, G, 3, 1]

    normal_map = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        # Black background for normals (background pixels → [0, 0, 0])
        torch.zeros((extrinsics.shape[0], 3), dtype=normals_per_gaussian.dtype, device=normals_per_gaussian.device),
        gaussian_means,
        gaussian_covariances,
        fake_sh,
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,  # <-- Critical: bypass SH evaluation, use colors_precomp
    )  # [B, 3, H, W]

    # L2-normalise per-pixel so the rendered normals are proper unit vectors.
    normal_map = torch.nn.functional.normalize(normal_map, dim=1, eps=1e-8)
    return normal_map


def render_depth_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    gaussian_means,
    gaussian_covariances,
    gaussian_opacities,
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
):
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "relative_disparity":
        fake_color = depth_to_relative_disparity(
            fake_color, near[:, None], far[:, None]
        )
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
    )
    return result.mean(dim=1)
