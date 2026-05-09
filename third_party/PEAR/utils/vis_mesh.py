import smplx
import trimesh
import torch
import numpy as np



smplx_model = smplx.SMPLX(
                model_path="./assets/SMPLX/SMPLX_NEUTRAL_2020.npz",
                gender="neutral",
                use_pca=False,
                flat_hand_mean=True,
            ).cuda()


def smpl_para2mesh(
    smpl_parameters: dict[str, torch.Tensor],
    batch_size,
    return_mesh: bool = True,
    verbose: bool = False,
) :
    """
    Convert SMPL parameters to meshes and vertices.

    Args:
        smpl_parameters: Dictionary containing SMPL parameters
        return_mesh: Whether to return meshes (default: True)

    Returns:
        Tuple of (list of meshes, numpy array of vertices)
    """
    faces = smplx_model.faces
    meshes = []
    smpl_vertices = []

    batch_parameters = get_batched_parameters(
        smpl_parameters, 0, batch_size, "cuda:0", "smpl"
    )

    with torch.no_grad():
        smpl_output = smplx_model(**batch_parameters)
        smpl_vertices.append(smpl_output.vertices.detach().cpu().numpy())

    smpl_vertices = np.concatenate(smpl_vertices, axis=0)
    if return_mesh:
        for vertices in smpl_vertices:
            meshes.append(trimesh.Trimesh(vertices, faces, process=False))

    return meshes, smpl_vertices


def get_batched_parameters(
    parameter_dict: dict[str, torch.Tensor],
    batch_start: int,
    batch_end: int,
    device: str,
    model_type: str = "mhr",
) -> dict[str, torch.Tensor]:
    """Get batched parameters for a specific range of frames.

    This is a generic parameter batching function that works with both MHR and SMPL(X) models.
    It extracts a batch slice from a parameter dictionary and handles parameter expansion
    for single-identity cases.

    Args:
        parameter_dict: Dictionary containing model parameters
        batch_start: Start index of the batch
        batch_end: End index of the batch
        device: Device to place the tensors on
        model_type: Type of model - "mhr" or "smpl" (default: "mhr")

    Returns:
        Dictionary containing batched parameters for the specified frame range
    """
    batched_parameter_dict = {}

    for k, v in parameter_dict.items():
        if v.shape[0] < batch_end:
            batched_parameter_dict[k] = v.expand(batch_end - batch_start, -1)
        else:
            batched_parameter_dict[k] = v[batch_start:batch_end].to(device)

    # For SMPLX model, ensure all required parameters exist with correct batch dimension
    if model_type == "smpl":
        batched_parameter_dict = complete_smplx_parameters(
            batched_parameter_dict, batch_end - batch_start, device
        )

    return batched_parameter_dict


def complete_smplx_parameters(
    smplx_parameters: dict[str, torch.Tensor],
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Complete SMPLX parameters with default values for missing keys.

    SMPLX models require additional parameters (jaw_pose, eye poses, hand poses,
    expression) that may not be present in the parameter dictionary. This function
    fills in missing parameters with zero tensors.

    Args:
        smplx_parameters: Dictionary of SMPLX parameters (may be incomplete)
        batch_size: Number of frames in the batch
        device: Device to place the tensors on

    Returns:
        Complete SMPLX parameter dictionary with all required keys
    """
    if "jaw_pose" not in smplx_parameters:
        smplx_parameters["jaw_pose"] = torch.zeros([batch_size, 1, 3], device=device)
    if "leye_pose" not in smplx_parameters:
        smplx_parameters["leye_pose"] = torch.zeros([batch_size, 1, 3], device=device)
    if "reye_pose" not in smplx_parameters:
        smplx_parameters["reye_pose"] = torch.zeros([batch_size, 1, 3], device=device)

    # Hand pose dimensions depend on PCA usage
    if "left_hand_pose" not in smplx_parameters:
        # Default to 6 dimensions (PCA mode), will be overridden if needed
        smplx_parameters["left_hand_pose"] = torch.zeros([batch_size, 6], device=device)
    if "right_hand_pose" not in smplx_parameters:
        smplx_parameters["right_hand_pose"] = torch.zeros(
            [batch_size, 6], device=device
        )

    # Expression parameters
    if "expression" not in smplx_parameters:
        # Default to 10 expression coefficients, will be overridden if needed
        smplx_parameters["expression"] = torch.zeros([batch_size, 10], device=device)

    return smplx_parameters
