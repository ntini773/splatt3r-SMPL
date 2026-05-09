

import torch
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, List
import torch
import numpy as np 
def perspective_projection(
    points         : torch.Tensor,
    translation    : torch.Tensor,
    focal_length   : torch.Tensor,
    camera_center  : Optional[torch.Tensor] = None,
    rotation       : Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Computes the perspective projection of a set of 3D points.
    https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/utils/geometry.py#L64-L102

    ### Args
        - points: torch.Tensor, (B, N, 3)
            - The input 3D points.
        - translation: torch.Tensor, (B, 3)
            - The 3D camera translation.
        - focal_length: torch.Tensor, (B, 2)
            - The focal length in pixels.
        - camera_center: torch.Tensor, (B, 2)
            - The camera center in pixels.
        - rotation: torch.Tensor, (B, 3, 3)
            - The camera rotation.

    ### Returns
        - torch.Tensor, (B, N, 2)
            - The projection of the input points.
    '''
    B = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(B, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(B, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([B, 3, 3], device=points.device, dtype=points.dtype)
    K[:,   0,  0] = focal_length[:, 0]
    K[:,   1,  1] = focal_length[:, 1]
    K[:,   2,  2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij, bkj -> bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij, bkj -> bki', K, projected_points)

    return projected_points[:, :, :-1]

def to_numpy(x):
    return x.detach().cpu().numpy()

def to_tensor(x, device, temporary:bool=False):
    '''
    Simply unify the type transformation to torch.Tensor. 
    If device is None, don't change the device if device is not CPU. 
    '''
    if isinstance(x, torch.Tensor):
        device = x.device if device is None else device
        if temporary:
            recover_type_back = lambda x_: x_.to(x.device)  # recover the device
            return x.to(device), recover_type_back
        else:
            return x.to(device)

    device = 'cpu' if device is None else device
    if isinstance(x, np.ndarray):
        if temporary:
            recover_type_back = lambda x_: x_.detach().cpu().numpy()
            return torch.from_numpy(x).to(device), recover_type_back
        else:
            return torch.from_numpy(x).to(device)
    if isinstance(x, List):
        if temporary:
            recover_type_back = lambda x_: x_.tolist()
            return torch.from_numpy(np.array(x)).to(device), recover_type_back
        else:
            return torch.from_numpy(np.array(x)).to(device)
    raise ValueError(f"Unsupported type: {type(x)}")

def visualize_keypoints_with_labels(keypoints: torch.Tensor, save_html: str = "keypoints_plot.html"):
    # 保证是 CPU 上的 numpy
    keypoints = keypoints.detach().cpu().numpy()
    x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]

    # 绘制点
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(size=4, color='red'),
        text=[str(i) for i in range(len(x))],  # 每个点的编号
        textposition="top center"
    )

    # 布局设置
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.write_html(save_html)
    print(f"[✔] 可视化保存为 HTML：{save_html}")

