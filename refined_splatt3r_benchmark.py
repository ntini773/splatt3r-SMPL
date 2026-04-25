import argparse
import json
import os
import sys

import einops
import lightning as L
import lpips
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

import utils.compute_ssim as compute_ssim
import utils.loss_mask as loss_mask
import workspace
from data.mvhumannet.mvhumannet import ALL_CAMERA_IDS
from src.anatomical_prior.anatomical_dataset import MVHumanNetAnatomicalData
from train_anatomical_refinement import MAST3RAnatomicalRefinement


class RefinedSplatt3RBenchmarkDataset(Dataset):
    def __init__(self, data, resolution, adj):
        super().__init__()
        self.data = data
        self.resolution = resolution
        self.adj = adj
        self.img_transform = torchvision.transforms.ToTensor()
        try:
            from src.mast3r_src.dust3r.dust3r.datasets.utils.transforms import ImgNorm
            self.enc_transform = ImgNorm
        except Exception as exc:
            raise ImportError('Could not import ImgNorm from the MASt3R source tree.') from exc

        self.index = [
            (seq, frame_name)
            for seq in self.data.sequences
            for frame_name in self.data.frame_names[seq]
        ]

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _depth_to_points(view):
        from src.mast3r_src.dust3r.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

        return depthmap_to_absolute_camera_coordinates(**view)

    def _load_context_view(self, seq, cam_id, frame_name):
        view = self.data.get_view(seq, cam_id, frame_name, self.resolution)
        view['img'] = self.enc_transform(view['original_img'])
        view['original_img'] = self.img_transform(view['original_img'])

        pts3d, valid_mask = self._depth_to_points(view)
        view['pts3d'] = pts3d
        view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
        view['mask_human'] = self.data.get_mask(seq, cam_id, frame_name, self.resolution)
        view['mesh_anchors'] = torch.from_numpy(self.data.get_anchors(seq, frame_name))
        return view

    def __getitem__(self, idx):
        seq, frame_name = self.index[idx]
        camera_seed = idx % len(ALL_CAMERA_IDS)

        context_indices = [camera_seed, (camera_seed + 2) % len(ALL_CAMERA_IDS)]
        target_indices = [
            (camera_seed - 1) % len(ALL_CAMERA_IDS),
            (camera_seed + 1) % len(ALL_CAMERA_IDS),
            (camera_seed + 3) % len(ALL_CAMERA_IDS),
        ]

        context = [
            self._load_context_view(seq, f'cam_{cam_idx:02d}', frame_name)
            for cam_idx in context_indices
        ]

        target = []
        for cam_idx in target_indices:
            cam_id = f'cam_{cam_idx:02d}'
            view = self.data.get_view(seq, cam_id, frame_name, self.resolution)
            view['original_img'] = self.img_transform(view['original_img'])
            view['mask_human'] = self.data.get_mask(seq, cam_id, frame_name, self.resolution)
            view['normal_map'] = self.data.get_normal(seq, cam_id, frame_name, self.resolution)
            target.append(view)

        return {
            'context': context,
            'target': target,
            'scene': seq,
            'frame_name': frame_name,
            'smpl_adj': self.adj,
            'smplx': self.data.get_smplx(seq, frame_name),
        }


def _make_dataset(config):
    dataset_root = getattr(config.data, 'root', '/ssd_scratch/gnrs/mvhumannet++_10demo')
    main_root = os.path.join(dataset_root, 'main')
    depth_root = os.path.join(dataset_root, 'depth')
    normal_root = os.path.join(dataset_root, 'normal')
    anchor_root = os.path.join(dataset_root, 'anchors')
    adj_path = os.path.join(dataset_root, 'smpl_adj_512.npy')
    fps_path = os.path.join(dataset_root, 'fps_indices_512.npy')
    smplx_model_path = getattr(config.data, 'smplx_model_path', 'smplx_models/')

    data = MVHumanNetAnatomicalData(
        main_root=main_root,
        depth_root=depth_root,
        normal_root=normal_root,
        anchor_root=anchor_root,
        smplx_model_path=smplx_model_path,
        fps_indices_path=fps_path,
        sequences=getattr(config.data, 'sequences', None),
    )
    adj = torch.from_numpy(np.load(adj_path).astype(np.float32))
    return RefinedSplatt3RBenchmarkDataset(data, config.data.resolution, adj)


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _save_visualizations(batch, predicted_color, target_color, rendered_normals, gt_normals, human_mask, soft_mask, dataset_human_mask, geom_mask, save_dir, step):
    if os.getenv('LOCAL_RANK', '0') != '0':
        return

    os.makedirs(save_dir, exist_ok=True)
    b = 0
    v_tgt = target_color.shape[1]
    img_list = []

    num_context = len(batch['context'])
    for v in range(v_tgt):
        if v < num_context:
            img_list.append(batch['context'][v]['original_img'][b].detach().cpu().clamp(0, 1))
        else:
            img_list.append(torch.zeros_like(target_color[b, v]).cpu())

    def get_rgb(tensor, v):
        return tensor[b, v].detach().cpu().clamp(0, 1)

    def to_3ch(tensor_mask, v):
        return tensor_mask[b, v].detach().cpu().float().unsqueeze(0).repeat(3, 1, 1)

    def get_masked_norm(tensor, mask_tensor, v):
        norm = tensor[b, v].detach().cpu() * 0.5 + 0.5
        mask_raw = mask_tensor[b, v].detach().cpu().unsqueeze(0)
        return norm * mask_raw

    for v in range(v_tgt):
        img_list.append(get_rgb(target_color, v))
    for v in range(v_tgt):
        img_list.append(get_rgb(predicted_color, v))
    for v in range(v_tgt):
        if gt_normals is not None:
            img_list.append(get_masked_norm(gt_normals, dataset_human_mask, v))
        else:
            img_list.append(torch.zeros_like(rendered_normals[b, v]).cpu())
    for v in range(v_tgt):
        img_list.append(get_masked_norm(rendered_normals, dataset_human_mask, v))
    for v in range(v_tgt):
        img_list.append(to_3ch(geom_mask, v))
    for v in range(v_tgt):
        img_list.append(to_3ch(dataset_human_mask, v))
    for v in range(v_tgt):
        img_list.append(to_3ch(human_mask, v))

    grid = torchvision.utils.make_grid(img_list, nrow=v_tgt)
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, f'step_{step:06d}.jpg')
    torchvision.utils.save_image(grid, save_path)
    print(f'Saved visualization to {save_path}')


@torch.no_grad()
def _evaluate_batch(model, batch, loss_weights, average_over_mask=True, background_weight=0.1):
    device = next(model.parameters()).device
    batch = _move_to_device(batch, device)

    _, _, h, w = batch['context'][0]['img'].shape
    view1, view2 = batch['context']
    adj = batch['smpl_adj'][0]
    anchors1 = view1['mesh_anchors']
    anchors2 = view2['mesh_anchors']

    pred1, pred2 = model.forward(view1, view2, adj, anchors1, anchors2)
    color, _ = model.decoder(batch, pred1, pred2, (h, w))
    rendered_normals = model.decoder.render_normals(batch, pred1, pred2, (h, w))

    mask = loss_mask.calculate_loss_mask(batch)
    target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
    predicted_color = color

    dataset_human_mask = torch.stack(
        [
            torch.from_numpy(t['mask_human']) if isinstance(t['mask_human'], np.ndarray) else t['mask_human']
            for t in batch['target']
        ],
        dim=1,
    ).to(device)

    final_mask = mask & dataset_human_mask
    if average_over_mask:
        inverted_mask = 1.0 - final_mask.float()
        eroded_inv = torch.nn.functional.max_pool2d(inverted_mask, kernel_size=7, stride=1, padding=3)
        loss_mask_tensor = 1.0 - eroded_inv
    else:
        loss_mask_tensor = final_mask.float()

    human_mask = loss_mask_tensor.float()
    soft_mask = human_mask + background_weight * (1.0 - human_mask)

    flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
    flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
    flattened_human_mask = einops.rearrange(human_mask, 'b v h w -> (b v) h w')

    rgb_l2_loss = (predicted_color - target_color) ** 2
    if average_over_mask:
        mse_loss = (rgb_l2_loss * soft_mask[:, :, None, :, :]).sum() / soft_mask.sum().clamp(min=1)
    else:
        mse_loss = (rgb_l2_loss * soft_mask[:, :, None, :, :]).mean()

    lpips_loss = model.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
    if average_over_mask:
        lpips_loss = (lpips_loss * flattened_human_mask[:, None, ...]).sum() / flattened_human_mask.sum().clamp(min=1)
    else:
        lpips_loss = lpips_loss.mean()

    ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
    if average_over_mask:
        ssim_val = (ssim_val * flattened_human_mask[:, None, ...]).sum() / flattened_human_mask.sum().clamp(min=1)
    else:
        ssim_val = ssim_val.mean()

    gt_normals = torch.stack([t['normal_map'] for t in batch['target']], dim=1).float().to(device)
    gt_normals = gt_normals * 2.0 - 1.0
    gt_normals = torch.nn.functional.normalize(gt_normals, dim=2, eps=1e-8)
    cos_sim = torch.nn.functional.cosine_similarity(rendered_normals, gt_normals, dim=2)
    normal_loss = (1.0 - cos_sim) * human_mask
    if average_over_mask:
        normal_loss = normal_loss.sum() / human_mask.sum().clamp(min=1)
    else:
        normal_loss = normal_loss.mean()

    loss = (
        loss_weights['mse'] * mse_loss
        + loss_weights['lpips'] * lpips_loss
        + loss_weights['normal'] * normal_loss
    )

    metrics = {
        'loss': loss.item(),
        'mse': mse_loss.item(),
        'psnr': (-10.0 * mse_loss.log10()).item(),
        'lpips': lpips_loss.item(),
        'ssim': ssim_val.item(),
        'normal': normal_loss.item(),
    }

    return metrics, color, target_color, rendered_normals, gt_normals, human_mask, soft_mask, dataset_human_mask, mask


def run_benchmark(config):
    L.seed_everything(getattr(config, 'seed', 42), workers=True)

    save_dir = getattr(config, 'eval_save_dir', '/ssd_scratch/gnrs/refined_splatt3r_eval_results')
    os.makedirs(save_dir, exist_ok=True)

    refined_ckpt = getattr(config, 'refined_checkpoint_path', None)
    if refined_ckpt is None:
        refined_ckpt = getattr(config, 'checkpoint_path', None)
    if refined_ckpt is None:
        raise ValueError('Provide config.refined_checkpoint_path (or config.checkpoint_path) for the refined checkpoint.')

    print(f'Loading refined model. Base ckpt: {config.splatt3r_checkpoint_path}')
    model = MAST3RAnatomicalRefinement(config)

    print(f'Loading refined checkpoint weights from {refined_ckpt}')
    ckpt = torch.load(refined_ckpt, map_location='cpu')
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'Warning: missing keys while loading refined checkpoint: {len(missing)}')
    if unexpected:
        print(f'Warning: unexpected keys while loading refined checkpoint: {len(unexpected)}')

    model.cuda().eval()
    model.lpips_criterion = lpips.LPIPS('vgg', spatial=True).cuda().eval()

    dataset = _make_dataset(config)
    data_loader = DataLoader(
        dataset,
        batch_size=getattr(config.data, 'batch_size', 1),
        shuffle=False,
        num_workers=getattr(config.data, 'num_workers', 0),
        pin_memory=True,
    )

    loss_weights = {
        'mse': getattr(config.loss, 'mse_loss_weight', 1.0),
        'lpips': getattr(config.loss, 'lpips_loss_weight', 1.0),
        'normal': getattr(config.loss, 'normal_loss_weight', 0.1),
    }
    average_over_mask = getattr(config.loss, 'average_over_mask', True)
    background_weight = getattr(config.loss, 'background_weight', 0.1)

    aggregate = {'loss': [], 'mse': [], 'psnr': [], 'lpips': [], 'ssim': [], 'normal': []}
    per_sample = {}

    for step, batch in enumerate(data_loader):
        metrics, color, target_color, rendered_normals, gt_normals, human_mask, soft_mask, dataset_human_mask, geom_mask = _evaluate_batch(
            model,
            batch,
            loss_weights=loss_weights,
            average_over_mask=average_over_mask,
            background_weight=background_weight,
        )

        scene = batch['scene'][0] if isinstance(batch['scene'], list) else batch['scene']
        frame_name = batch.get('frame_name', ['sample'])[0] if isinstance(batch.get('frame_name', ['sample']), list) else batch.get('frame_name', 'sample')
        sample_key = f'{scene}/{frame_name}/{step:06d}'
        per_sample[sample_key] = metrics

        for key in aggregate:
            aggregate[key].append(metrics[key])

        if step < 10:
            _save_visualizations(
                batch=batch,
                predicted_color=color,
                target_color=target_color,
                rendered_normals=rendered_normals,
                gt_normals=gt_normals,
                human_mask=human_mask,
                soft_mask=soft_mask,
                dataset_human_mask=dataset_human_mask,
                geom_mask=geom_mask,
                save_dir=save_dir,
                step=step,
            )

        print(
            f"[{step + 1:04d}/{len(data_loader):04d}] {sample_key}: "
            f"loss={metrics['loss']:.4f} mse={metrics['mse']:.4f} psnr={metrics['psnr']:.2f} "
            f"lpips={metrics['lpips']:.4f} ssim={metrics['ssim']:.4f} normal={metrics['normal']:.4f}"
        )

    summary = {key: float(np.mean(values)) for key, values in aggregate.items()}
    print('\nRefined Splatt3R evaluation summary:')
    for key, value in summary.items():
        print(f'  {key}: {value:.6f}')

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'per_sample': per_sample}, f, indent=2)

    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f'{key}: {value:.6f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark a refined anatomical Splatt3R checkpoint on MVHumanNet.')
    parser.add_argument('config_path', help='Path to YAML config used for data paths and evaluation settings.')
    parser.add_argument('overrides', nargs=argparse.REMAINDER, help='Optional config overrides passed to workspace.load_config.')
    args = parser.parse_args()

    config = workspace.load_config(args.config_path, args.overrides)
    run_benchmark(config)