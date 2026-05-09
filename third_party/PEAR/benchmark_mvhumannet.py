import argparse
import os
import torch
import numpy as np
import cv2
import trimesh
from tqdm import tqdm
import smplx
import lightning
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from models.modules.ehm import EHM_v2
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import ConfigDict, add_extra_cfgs

# Metric functions
def rigid_alignment(A, B):
    """
    Rigid alignment of two point clouds (A to B) using Umeyama's algorithm.
    Returns (aligned_A, B, error)
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    H = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_B - R @ centroid_A
    A_aligned = (R @ A.T).T + t
    
    return A_aligned, B, np.mean(np.linalg.norm(A_aligned - B, axis=-1))

def compute_pa_mpjpe(pred_joints, gt_joints):
    """Procrustes-aligned MPJPE."""
    B, J, C = pred_joints.shape
    errors = []
    
    pred_np = pred_joints.detach().cpu().numpy()
    gt_np = gt_joints.detach().cpu().numpy()
    
    for i in range(B):
        _, _, error = rigid_alignment(pred_np[i], gt_np[i])
        errors.append(error)
        
    return np.mean(errors)

def load_img(path, order="RGB", scale=1):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError(f"Fail to read {path}")

    if order == "RGB":
        img = img[:, :, ::-1].copy()

    if scale != 1:
        height, width = img.shape[:2]
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

    return img.astype(np.float32)

def select_person_bbox(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
    areas = np.maximum(0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
    score = conf * areas
    return xyxy[int(np.argmax(score))]

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        return np.array([x1, y1, x2 - x1, y2 - y1])
    return None

def process_bbox(bbox, img_width, img_height, input_img_shape, ratio=1.25):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return None
    width = bbox[2]
    height = bbox[3]
    center_x = bbox[0] + width / 2.0
    center_y = bbox[1] + height / 2.0
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if width > aspect_ratio * height:
        height = width / aspect_ratio
    elif width < aspect_ratio * height:
        width = height * aspect_ratio
    bbox[2] = width * ratio
    bbox[3] = height * ratio
    bbox[0] = center_x - bbox[2] / 2.0
    bbox[1] = center_y - bbox[3] / 2.0
    return bbox.astype(np.float32)

def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, _ = img.shape
    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    
    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    return img_patch.astype(np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)
    src = np.array([src_center, src_center + src_downdir, src_center + src_rightdir], dtype=np.float32)
    dst = np.array([dst_center, dst_center + dst_downdir, dst_center + dst_rightdir], dtype=np.float32)
    return cv2.getAffineTransform(src, dst).astype(np.float32)

def rotate_2d(pt_2d, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([pt_2d[0] * cs - pt_2d[1] * sn, pt_2d[0] * sn + pt_2d[1] * cs], dtype=np.float32)

def load_models(repo_root, config_name, device, yolo_model_path):
    meta_cfg = ConfigDict(model_config_path=os.path.join(repo_root, "configs", f"{config_name}.yaml"))
    meta_cfg = add_extra_cfgs(meta_cfg)
    ehm_model = Ehm_Pipeline(meta_cfg)
    hub_file = hf_hub_download(repo_id="BestWJH/PEAR_models", filename="ehm_model_stage1.pt", repo_type="model")
    state = torch.load(hub_file, map_location="cpu", weights_only=True)
    ehm_model.backbone.load_state_dict(state["backbone"], strict=False)
    ehm_model.head.load_state_dict(state["head"], strict=False)
    ehm_model = ehm_model.to(device).eval()
    ehm = EHM_v2(os.path.join(repo_root, "assets", "FLAME"), os.path.join(repo_root, "assets", "SMPLX")).to(device).eval()
    detector = YOLO(yolo_model_path)
    return ehm_model, ehm, detector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--config_name", default="infer", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--yolo_model", default="./model_zoo/yolov8x.pt", type=str)
    parser.add_argument("--limit", type=int, default=100, help="limit number of frames for benchmark")
    parser.add_argument("--save_meshes", type=int, default=5, help="number of mesh pairs to save (0 to disable)")
    parser.add_argument("--mesh_out_dir", type=str, default="./debug_meshes", help="directory to save debug meshes")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    device = args.device
    ehm_model, ehm, detector = load_models(repo_root, args.config_name, device, args.yolo_model)
    
    # Official SMPL-X model for GT loading
    gt_smplx_model = smplx.SMPLX(
        model_path="/ssd_scratch/gnrs/checkpoints/smplx/",
        gender='neutral',
        use_pca=True,
        num_pca_comps=6,
        flat_hand_mean=True,
        batch_size=1
    ).to(device).eval()

    # Get GT faces for mesh saving
    gt_faces = gt_smplx_model.faces.astype(np.int32)
    # PEAR faces (with teeth)
    pred_faces = ehm.smplx.faces_tensor.cpu().numpy().astype(np.int32)

    if args.save_meshes > 0:
        os.makedirs(args.mesh_out_dir, exist_ok=True)
        print(f"Will save {args.save_meshes} mesh pairs to {args.mesh_out_dir}")
    meshes_saved = 0

    main_root = os.path.join(args.dataset_root, "main")
    sequences = sorted([s for s in os.listdir(main_root) if os.path.isdir(os.path.join(main_root, s))])
    
    metrics = {
        "mpjpe": [],
        "pa_mpjpe": [],
        "mpvpe": []
    }
    
    transform = transforms.ToTensor()
    processed_count = 0

    for seq in sequences:
        if processed_count >= args.limit:
            break
            
        smplx_gt_path = os.path.join(main_root, seq, "smplx_params.npz")
        if not os.path.exists(smplx_gt_path):
            print(f"[skip] no smplx GT for {seq}")
            continue
            
        smplx_gt_data = np.load(smplx_gt_path)
        
        cam_dir = os.path.join(main_root, seq, "images", "cam_00")
        frames = sorted([f for f in os.listdir(cam_dir) if f.endswith(".jpg") and not f.startswith("A-")])
        
        for frame_file in tqdm(frames, desc=f"Seq {seq}"):
            if processed_count >= args.limit:
                break
                
            frame_id_str = os.path.splitext(frame_file)[0]
            try:
                # MVHumanNet++ index logic: smplx_id = int(frame_id) // 25 - 1
                smplx_id = int(frame_id_str) // 25 - 1
                if smplx_id < 0 or smplx_id >= len(smplx_gt_data['body_pose']):
                    continue
            except:
                continue

            img_path = os.path.join(cam_dir, frame_file)
            image = load_img(img_path)
            h, w = image.shape[:2]
            
            # 1. Prediction
            detections = detector.predict(image, device=device, classes=0, conf=0.5, save=False, verbose=False)[0]
            bbox = select_person_bbox(detections.boxes)
            if bbox is None: continue
            
            bbox_xywh = np.array([bbox[0], bbox[1], abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1])], dtype=np.float32)
            bbox_xywh = process_bbox(bbox_xywh, w, h, [256, 256], ratio=1.25)
            if bbox_xywh is None: continue
            
            img_patch = generate_patch_image(image, bbox_xywh, scale=1.0, rot=0.0, do_flip=False, out_shape=[256, 256])
            img_patch_tensor = transform(img_patch.astype(np.float32)) / 255.0
            img_patch_tensor = img_patch_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_outputs = ehm_model(img_patch_tensor)
                # Zero out global orientation for pred (camera frame ≠ GT world frame)
                pred_body_param = pred_outputs["body_param"].copy() if isinstance(pred_outputs["body_param"], dict) else {k: v for k, v in pred_outputs["body_param"].items()}
                pred_body_param["global_pose"] = torch.zeros_like(pred_body_param["global_pose"])
                pred_smplx = ehm(pred_body_param, pred_outputs["flame_param"], pose_type="aa")
                
                # 2. Ground Truth — zero global_orient and transl for canonical comparison
                gt_params = {
                    "betas": torch.from_numpy(smplx_gt_data['betas'][0][None]).to(device).float(),
                    "global_orient": torch.zeros(1, 3, device=device),  # zeroed
                    "body_pose": torch.from_numpy(smplx_gt_data['body_pose'][smplx_id][None]).to(device).float(),
                    "left_hand_pose": torch.from_numpy(smplx_gt_data['left_hand_pose'][smplx_id][None]).to(device).float(),
                    "right_hand_pose": torch.from_numpy(smplx_gt_data['right_hand_pose'][smplx_id][None]).to(device).float(),
                }
                
                gt_output = gt_smplx_model(**gt_params)
                
                # 3. Align and Compare
                # PEAR EHM_v2 outputs 55 joints
                pred_joints = pred_smplx["joints"][:, :55, :]
                gt_joints = gt_output.joints[:, :55, :]
                
                # PEAR EHM adds 120 teeth verts (10595 total), official smplx has 10475
                pred_verts = pred_smplx["vertices"][:, :gt_output.vertices.shape[1], :]
                gt_verts = gt_output.vertices
                
                # Root alignment (Joint 0 is pelvis)
                pred_root = pred_joints[:, :1, :]
                gt_root = gt_joints[:, :1, :]
                
                pred_joints_centered = pred_joints - pred_root
                gt_joints_centered = gt_joints - gt_root
                pred_verts_centered = pred_verts - pred_root
                gt_verts_centered = gt_verts - gt_root
                
                mpjpe = torch.mean(torch.norm(pred_joints_centered - gt_joints_centered, dim=-1)).item()
                mpvpe = torch.mean(torch.norm(pred_verts_centered - gt_verts_centered, dim=-1)).item()
                pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints)
                
                metrics["mpjpe"].append(mpjpe)
                metrics["mpvpe"].append(mpvpe)
                metrics["pa_mpjpe"].append(pa_mpjpe)
                
                # Save debug meshes for visual inspection
                if meshes_saved < args.save_meshes:
                    tag = f"{seq}_{frame_id_str}"
                    # Save root-centered meshes
                    pv = pred_verts_centered[0].cpu().numpy()
                    gv = gt_verts_centered[0].cpu().numpy()
                    # Pred mesh (truncated to 10475 verts → use GT faces)
                    trimesh.Trimesh(vertices=pv, faces=gt_faces, process=False).export(
                        os.path.join(args.mesh_out_dir, f"pred_{tag}.obj"))
                    trimesh.Trimesh(vertices=gv, faces=gt_faces, process=False).export(
                        os.path.join(args.mesh_out_dir, f"gt_{tag}.obj"))
                    # Also save the input image patch for reference
                    cv2.imwrite(os.path.join(args.mesh_out_dir, f"input_{tag}.jpg"),
                                img_patch[:, :, ::-1])  # RGB→BGR
                    meshes_saved += 1
                    print(f"  [saved mesh pair {meshes_saved}/{args.save_meshes}: {tag}]")
                
                processed_count += 1

    if processed_count > 0:
        print("\n" + "="*30)
        print("BENCHMARK RESULTS")
        print("="*30)
        print(f"Frames processed: {processed_count}")
        print(f"MPJPE:    {np.mean(metrics['mpjpe'])*1000:7.2f} mm")
        print(f"PA-MPJPE: {np.mean(metrics['pa_mpjpe'])*1000:7.2f} mm")
        print(f"MPVPE:    {np.mean(metrics['mpvpe'])*1000:7.2f} mm")
        print("="*30)
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    main()
