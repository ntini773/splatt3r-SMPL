import argparse
import os
import multiprocessing as mp

import cv2
import lightning
import numpy as np
import torch
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from models.modules.ehm import EHM_v2
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import ConfigDict, add_extra_cfgs


def load_img(path, order="RGB", scale=2):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError(f"Fail to read {path}")

    if order == "RGB":
        img = img[:, :, ::-1].copy()

    if scale != 1:
        height, width = img.shape[:2]
        img = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

    return img.astype(np.float32)


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


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans.astype(np.float32)


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
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True
    )

    return img_patch, trans, inv_trans


def build_cameras_kwargs(batch_size, focal_length):
    screen_size = torch.tensor([1024, 1024]).float()[None].repeat(batch_size, 1)
    return {
        "principal_point": torch.zeros(batch_size, 2).float(),
        "focal_length": focal_length,
        "image_size": screen_size,
        "device": "cuda",
    }


def farthest_point_sampling(points, n_samples):
    num_points = len(points)
    selected = np.zeros(n_samples, dtype=int)
    distances = np.full(num_points, np.inf)
    selected[0] = 0
    for i in range(1, n_samples):
        last_point = points[selected[i - 1]]
        dist = np.sum((points - last_point) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)
    return selected


def build_adjacency(fps_indices, smplx_faces, n_nodes=512):
    idx_map = {vertex: idx for idx, vertex in enumerate(fps_indices.tolist())}
    fps_set = set(fps_indices.tolist())
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for face in smplx_faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[0], face[2])):
            if a in fps_set and b in fps_set:
                ia = idx_map[a]
                ib = idx_map[b]
                adj[ia, ib] = 1.0
                adj[ib, ia] = 1.0
    np.fill_diagonal(adj, 1.0)
    degree = adj.sum(axis=1).clip(min=1)
    adj = adj / np.sqrt(degree[:, None]) / np.sqrt(degree[None, :])
    return adj


def save_obj(mesh_path, vertices, faces=None):
    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
    vertices = np.asarray(vertices)
    faces = None if faces is None else np.asarray(faces)

    with open(mesh_path, "w") as f:
        f.write("# generated by preprocess_mvhumannet_anchors.py\n")
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        if faces is not None:
            for i, j, k in faces:
                f.write(f"f {int(i) + 1} {int(j) + 1} {int(k) + 1}\n")


def build_reduced_mesh(fps_indices, faces):
    fps_map = {int(vertex): idx for idx, vertex in enumerate(fps_indices.tolist())}
    reduced_faces = []
    for face in np.asarray(faces):
        if all(int(v) in fps_map for v in face):
            reduced_faces.append([fps_map[int(face[0])], fps_map[int(face[1])], fps_map[int(face[2])]])

    if len(reduced_faces) == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(reduced_faces, dtype=np.int32)


def build_knn_faces(vertices, k_neighbors=6):
    num_vertices = vertices.shape[0]
    if num_vertices < 3:
        return np.empty((0, 3), dtype=np.int32)

    distances = np.sum((vertices[:, None, :] - vertices[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :k_neighbors]

    face_set = set()
    for vertex_id in range(num_vertices):
        local = neighbors[vertex_id]
        local_count = len(local)
        for i in range(local_count):
            for j in range(i + 1, local_count):
                a = int(vertex_id)
                b = int(local[i])
                c = int(local[j])
                if a != b and b != c and a != c:
                    face = tuple(sorted((a, b, c)))
                    face_set.add(face)

    if not face_set:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(sorted(face_set), dtype=np.int32)


def generate_global_files(repo_root, output_root, n_anchors=512):
    fps_path = os.path.join(output_root, f"fps_indices_{n_anchors}.npy")
    adj_path = os.path.join(output_root, f"smpl_adj_{n_anchors}.npy")

    if os.path.exists(fps_path) and os.path.exists(adj_path):
        print(f"[skip] found {fps_path} and {adj_path}")
        return np.load(fps_path).astype(int)

    from models.modules.ehm import EHM_v2

    print("Generating global anchor metadata...")
    ehm = EHM_v2(os.path.join(repo_root, "assets", "FLAME"), os.path.join(repo_root, "assets", "SMPLX"))
    rest_vertices = ehm.smplx.v_template.detach().cpu().numpy()
    smplx_faces = ehm.smplx.faces_tensor.detach().cpu().numpy()

    fps_indices = farthest_point_sampling(rest_vertices, n_anchors)
    np.save(fps_path, fps_indices)
    print(f"saved {fps_path}")

    adj = build_adjacency(fps_indices, smplx_faces, n_nodes=n_anchors)
    np.save(adj_path, adj)
    print(f"saved {adj_path}")

    return fps_indices


def select_person_bbox(boxes):
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
    areas = np.maximum(0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
    score = conf * areas
    return xyxy[int(np.argmax(score))]


def collect_frame_images(dataset_root):
    main_root = os.path.join(dataset_root, "main")
    sequences = sorted([seq for seq in os.listdir(main_root) if os.path.isdir(os.path.join(main_root, seq))])

    frame_items = []
    for seq in sequences:
        cam_dir = os.path.join(main_root, seq, "images", "cam_00")
        if not os.path.isdir(cam_dir):
            continue

        frames = sorted(
            [name for name in os.listdir(cam_dir) if name.lower().endswith((".jpg", ".jpeg", ".png")) and not name.startswith("A-")]
        )
        for frame_file in frames:
            frame_name = os.path.splitext(frame_file)[0]
            frame_items.append((seq, frame_name, os.path.join(cam_dir, frame_file)))

    return frame_items


def load_models(repo_root, config_name, device, yolo_model_path):
    meta_cfg = ConfigDict(model_config_path=os.path.join(repo_root, "configs", f"{config_name}.yaml"))
    meta_cfg = add_extra_cfgs(meta_cfg)
    lightning.fabric.seed_everything(10)

    ehm_model = Ehm_Pipeline(meta_cfg)
    hub_file = hf_hub_download(repo_id="BestWJH/PEAR_models", filename="ehm_model_stage1.pt", repo_type="model")
    state = torch.load(hub_file, map_location="cpu", weights_only=True)
    ehm_model.backbone.load_state_dict(state["backbone"], strict=False)
    ehm_model.head.load_state_dict(state["head"], strict=False)
    ehm_model = ehm_model.to(device).eval()

    ehm = EHM_v2(os.path.join(repo_root, "assets", "FLAME"), os.path.join(repo_root, "assets", "SMPLX")).to(device).eval()
    if not os.path.isabs(yolo_model_path):
        yolo_model_path = os.path.join(repo_root, yolo_model_path)
    detector = YOLO(yolo_model_path)

    return ehm_model, ehm, detector


def process_frame_items(
    frame_items,
    repo_root,
    output_root,
    config_name,
    device,
    yolo_model_path,
    fps_indices,
    conf_threshold,
    debug=False,
):
    anchor_root = os.path.join(output_root, "anchors")
    debug_root = os.path.join(output_root, "debug") if debug else None
    if debug_root is not None:
        os.makedirs(debug_root, exist_ok=True)

    ehm_model, ehm, detector = load_models(repo_root, config_name, device, yolo_model_path)
    template_faces = ehm.smplx.faces_tensor.detach().cpu().numpy()
    reduced_faces = build_reduced_mesh(fps_indices, template_faces)

    transform = transforms.ToTensor()
    print(f"[{device}] processing {len(frame_items)} frame entries")
    saved_count = 0
    skipped_count = 0

    for seq, frame_name, image_path in frame_items:
        anchor_path = os.path.join(anchor_root, seq, f"{frame_name}.npz")
        if os.path.exists(anchor_path):
            skipped_count += 1
            continue

        image = load_img(image_path)
        image_height, image_width = image.shape[:2]

        detections = detector.predict(
            image,
            device=device,
            classes=0,
            conf=conf_threshold,
            save=False,
            verbose=False,
        )[0]

        bbox = select_person_bbox(detections.boxes)
        if bbox is None:
            print(f"[skip] no person found: {seq}/{frame_name}")
            continue

        bbox_xywh = np.array(
            [bbox[0], bbox[1], abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])],
            dtype=np.float32,
        )
        bbox_xywh = process_bbox(bbox_xywh, image_width, image_height, input_img_shape=[256, 256], ratio=1.25)
        if bbox_xywh is None:
            print(f"[skip] invalid bbox: {seq}/{frame_name}")
            continue

        img_patch, _, _ = generate_patch_image(image, bbox_xywh, scale=1.0, rot=0.0, do_flip=False, out_shape=[256, 256])
        img_patch = transform(img_patch.astype(np.float32)) / 255.0
        img_patch = img_patch.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = ehm_model(img_patch)
            pd_smplx_dict = ehm(outputs["body_param"], outputs["flame_param"], pose_type="aa")

        mesh_vertices = pd_smplx_dict["vertices"][0].detach().cpu().numpy().astype(np.float32)
        anchors = mesh_vertices[fps_indices].astype(np.float32)

        seq_anchor_dir = os.path.join(anchor_root, seq)
        os.makedirs(seq_anchor_dir, exist_ok=True)
        np.savez_compressed(anchor_path, anchors=anchors)

        if debug:
            debug_seq_dir = os.path.join(debug_root, seq)
            full_obj_path = os.path.join(debug_seq_dir, f"{frame_name}_person.obj")
            reduced_obj_path = os.path.join(debug_seq_dir, f"{frame_name}_anchors.obj")
            reduced_mesh_path = os.path.join(debug_seq_dir, f"{frame_name}_reduced_mesh.obj")

            save_obj(full_obj_path, mesh_vertices, template_faces)
            save_obj(reduced_obj_path, anchors, None)

            if len(reduced_faces) > 0:
                connected_faces = reduced_faces
            else:
                connected_faces = build_knn_faces(anchors, k_neighbors=6)
            save_obj(reduced_mesh_path, anchors, connected_faces)

        saved_count += 1
        print(f"saved {anchor_path}")

    print(f"[{device}] done. saved={saved_count}, skipped={skipped_count}")


def _worker_run(
    frame_items,
    repo_root,
    output_root,
    config_name,
    device,
    yolo_model_path,
    fps_indices,
    conf_threshold,
    debug,
):
    process_frame_items(
        frame_items=frame_items,
        repo_root=repo_root,
        output_root=output_root,
        config_name=config_name,
        device=device,
        yolo_model_path=yolo_model_path,
        fps_indices=fps_indices,
        conf_threshold=conf_threshold,
        debug=debug,
    )


def preprocess(dataset_root, output_root, config_name, devices, yolo_model_path, n_anchors, conf_threshold, debug=False, debug_max_images=5):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_root, exist_ok=True)
    anchor_root = os.path.join(output_root, "anchors")
    os.makedirs(anchor_root, exist_ok=True)

    fps_indices = generate_global_files(repo_root, output_root, n_anchors=n_anchors)
    frame_items = collect_frame_images(dataset_root)
    if debug:
        frame_items = frame_items[:debug_max_images]
        print(f"[debug] limiting preprocessing to first {len(frame_items)} frames")

    if not frame_items:
        print("No frames found to process.")
        return

    gpu_ids = [token.strip() for token in str(devices).split(",") if token.strip() != ""]
    if len(gpu_ids) <= 1:
        device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda:0"
        process_frame_items(
            frame_items=frame_items,
            repo_root=repo_root,
            output_root=output_root,
            config_name=config_name,
            device=device,
            yolo_model_path=yolo_model_path,
            fps_indices=fps_indices,
            conf_threshold=conf_threshold,
            debug=debug,
        )
        return

    print(f"Using multi-GPU workers on devices: {gpu_ids}")
    frame_splits = [frame_items[idx::len(gpu_ids)] for idx in range(len(gpu_ids))]

    ctx = mp.get_context("spawn")
    workers = []
    for split_idx, gpu_id in enumerate(gpu_ids):
        device = f"cuda:{gpu_id}"
        worker = ctx.Process(
            target=_worker_run,
            args=(
                frame_splits[split_idx],
                repo_root,
                output_root,
                config_name,
                device,
                yolo_model_path,
                fps_indices,
                conf_threshold,
                debug,
            ),
        )
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

    failed = [worker.exitcode for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"One or more workers failed with exit codes: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--config_name", default="infer", type=str)
    parser.add_argument("--devices", default="0", type=str, help="single GPU '0' or multiple GPUs '0,1,2'")
    parser.add_argument("--yolo_model", default="./model_zoo/yolov8x.pt", type=str)
    parser.add_argument("--n_anchors", default=512, type=int)
    parser.add_argument("--conf_threshold", default=0.5, type=float)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root if args.output_root is not None else os.path.join(args.dataset_root, "processed_root")
    preprocess(
        dataset_root=args.dataset_root,
        output_root=output_root,
        config_name=args.config_name,
        devices=args.devices,
        yolo_model_path=args.yolo_model,
        n_anchors=args.n_anchors,
        conf_threshold=args.conf_threshold,
        debug=args.debug,
        debug_max_images=5,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()