import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
import sys
import torch
import einops
import lightning as L
import lpips
import omegaconf

sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.geometry as geometry
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils
import workspace
import numpy as np

import main as splatt3r_main  # gives us MAST3RGaussians with full Lightning checkpoint support
from src.anatomical_prior.gcn_encoder import AnatomicalGCNEncoder
from src.anatomical_prior.anatomical_dataset import get_anatomical_dataset

class MAST3RAnatomicalRefinement(L.LightningModule):
    """
    Wraps a pretrained Splatt3R checkpoint and adds a GCN anatomical prior
    via Cross-Attention inside the GaussianHead.

    CHECKPOINT LOADING STRATEGY
    ---------------------------
    We use MAST3RGaussians.load_from_checkpoint() (the Lightning API) — NOT
    a raw load_state_dict on a manually constructed AsymmetricMASt3R.
    This guarantees:
      1. All pretrained Gaussian DPT weights are loaded under the correct
         key hierarchy (encoder.downstream_head1.* etc.).
      2. The newly added prior_attention + norm layers are simply absent
         from the checkpoint; strict=False handles them gracefully.
      3. The PixelSplat splatting decoder and all hyperparameters are also
         correctly restored, consistent with how demo.py loads the model.

    WHAT IS TRAINABLE
    -----------------
      - gcn_encoder                            : freshly initialised
      - downstream_head1/2.prior_attention     : freshly initialised, zero out_proj
      - downstream_head1/2.gaussian_dpt.dpt   : pretrained, finetuned at low LR
    Everything else is frozen (ViT backbone, cross-attn decoder, local-feature MLP).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # background_weight scheduling state (updated in on_train_epoch_start)
        self._current_bg_weight = getattr(config.loss, 'background_weight', 0.1)

        # ------------------------------------------------------------------
        # 1. Load the full Splatt3R model via Lightning API so that ALL sub-
        #    module weights (encoder ViT, decoder, Gaussian DPT heads, splat
        #    decoder) are correctly restored from the checkpoint.
        # ------------------------------------------------------------------
        print(f"Loading Splatt3R checkpoint from {config.splatt3r_checkpoint_path} ...")
        splatt3r = splatt3r_main.MAST3RGaussians.load_from_checkpoint(
            config.splatt3r_checkpoint_path,
            map_location='cpu',
            strict=False,  # <--- MUST be False so it ignores the new attention/norm layers
        )

        # Expose encoder and decoder as top-level attributes so Lightning DDP
        # wraps them correctly.
        self.encoder = splatt3r.encoder
        self.decoder = splatt3r.decoder

        # ------------------------------------------------------------------
        # 2. Freeze everything in the Splatt3R backbone.
        # ------------------------------------------------------------------
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)

        # ------------------------------------------------------------------
        # 3. Selectively unfreeze what we want to finetune:
        #      a) The Gaussian DPT regression sub-network (low LR)
        #      b) The newly injected prior_attention + norm (freshly init)
        # ------------------------------------------------------------------
        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head1.prior_attention.requires_grad_(True)
        self.encoder.downstream_head2.prior_attention.requires_grad_(True)

        # ------------------------------------------------------------------
        # 4. Initialise the new GCN encoder (will be trained from scratch).
        # ------------------------------------------------------------------
        self.gcn_encoder = AnatomicalGCNEncoder(in_channels=3, hidden_channels=128, out_channels=32)

        # LPIPS for photometric loss
        if config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

        # Optional MASt3R geometry loss (unfreezes entire downstream heads)
        if getattr(config.loss, 'mast3r_loss_weight', None) is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        self.save_hyperparameters()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, view1, view2, adj, anchors1, anchors2):
        # Frozen backbone — no gradient graph needed here
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = \
                self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # GCN encoder is trainable — stays in the autograd graph
        prior1 = self.gcn_encoder(anchors1, adj)   # [B, 512, 32]
        prior2 = self.gcn_encoder(anchors2, adj)   # [B, 512, 32]

        # downstream_head calls the modified GaussianHead.forward which
        # accepts human_prior_features for cross-attention injection.
        pred1 = self.encoder.downstream_head1(
            [tok.float() for tok in dec1], shape1, human_prior_features=prior1
        )
        pred2 = self.encoder.downstream_head2(
            [tok.float() for tok in dec2], shape2, human_prior_features=prior2
        )

        # Clamp scales to prevent Gaussian blowup (which causes massive tile overlap and OOMs)
        pred1['scales'] = torch.clamp(pred1['scales'], max=0.05)
        pred2['scales'] = torch.clamp(pred2['scales'], max=0.05)

        pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])

        # Residual SH from image colors (Match base splatt3r shape logic)
        new_sh1 = torch.zeros_like(pred1['sh'])
        new_sh2 = torch.zeros_like(pred2['sh'])
        new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
        new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
        pred1['sh'] = pred1['sh'] + new_sh1
        pred2['sh'] = pred2['sh'] + new_sh2

        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')

        return pred1, pred2

    def on_train_epoch_start(self):
        """Update background_weight according to linear decay schedule."""
        loss_cfg = self.config.loss
        initial_bg  = getattr(loss_cfg, 'background_weight', 0.1)
        end_epoch   = getattr(loss_cfg, 'bg_schedule_end_epoch', None)
        psnr_gate   = getattr(loss_cfg, 'bg_schedule_psnr_gate', None)

        if end_epoch is None or self.current_epoch == 0:
            # No schedule configured, or not started yet
            self._current_bg_weight = initial_bg
        else:
            # Optionally gate: don't start decaying until PSNR reaches threshold
            psnr_ok = True
            if psnr_gate is not None:
                logged_psnr = self.trainer.logged_metrics.get('train/psnr', None)
                psnr_ok = (logged_psnr is not None) and (logged_psnr >= psnr_gate)

            if psnr_ok:
                # Linear decay: initial_bg → floor over [0, end_epoch]
                floor = getattr(loss_cfg, 'bg_weight_floor', 0.05)
                fraction = min(1.0, self.current_epoch / end_epoch)
                decayed = initial_bg * (1.0 - fraction)
                self._current_bg_weight = max(decayed, floor)
            else:
                self._current_bg_weight = initial_bg

        self.log('train/bg_weight', self._current_bg_weight, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        adj = batch['smpl_adj'][0]
        anchors1 = view1['mesh_anchors'].cuda()
        anchors2 = view2['mesh_anchors'].cuda()

        pred1, pred2 = self.forward(view1, view2, adj, anchors1, anchors2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        mask = loss_mask.calculate_loss_mask(batch)

        target_mask_human = torch.stack(
            [torch.from_numpy(t['mask_human']) if isinstance(t['mask_human'], np.ndarray) else t['mask_human'] 
             for t in batch['target']], dim=1
        ).to(mask.device)

        loss, mse, lpips_val = self.calculate_loss(
            batch, pred1, pred2, color, mask, target_mask_human,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
        )

        self.log_metrics('train', loss, mse, lpips_val)
        return loss

    def calculate_loss(self, batch, pred1, pred2, color, geom_mask, dataset_human_mask, apply_mask=True, average_over_mask=True):
        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color

        final_mask = geom_mask & dataset_human_mask

        if apply_mask:
            # Apply 3px erosion to the combined depth-derived + dataset human mask.
            # Erosion on a binary mask = 1 - MaxPool(1 - mask).
            inverted_mask = 1.0 - final_mask.float()
            eroded_inv = torch.nn.functional.max_pool2d(
                inverted_mask, kernel_size=7, stride=1, padding=3
            )
            mask = 1.0 - eroded_inv  # [B, V, H, W]
        else:
            mask = final_mask.float()

        # ── Three-zone soft mask ────────────────────────────────────────────
        # Zone 1: Inner human (eroded intersection mask)          weight = 1.0
        # Zone 2: Human boundary ring (silhouette minus erosion)  weight = 0.5 fixed
        #          This protects real human pixels that fall outside the strict
        #          geom intersection — they always keep a gradient signal.
        # Zone 3: Background (everything outside silhouette)      weight = bg_weight
        #          Decays via schedule but floors at bg_weight_floor to avoid
        #          opacity collapse from complete zero-gradient regions.
        #
        # Note: LPIPS is still human-only (inner zone 1 only) — unchanged.
        background_weight = self._current_bg_weight
        boundary_weight = getattr(self.config.loss, 'boundary_weight', 0.5)

        inner_human  = mask.float()                           # eroded intersection [B,V,H,W]
        human_sil    = dataset_human_mask.float()             # full dataset silhouette
        boundary_ring = (human_sil - inner_human).clamp(0, 1) # ring between silhouette & eroded core
        background    = 1.0 - human_sil                       # everything outside silhouette

        soft_mask = (
            inner_human * 1.0
            + boundary_ring * boundary_weight
            + background * background_weight
        )  # [B, V, H, W]

        # human_mask for LPIPS: use inner eroded core only (strict)
        human_mask = inner_human

        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_human_mask = einops.rearrange(human_mask, 'b v h w -> (b v) h w')

        # --- MSE Loss (soft-masked) ---
        rgb_l2_loss = (predicted_color - target_color) ** 2  # [B, V, C, H, W]
        if average_over_mask:
            mse_loss = (rgb_l2_loss * soft_mask[:, :, None, :, :]).sum() / (soft_mask.sum() + 1e-8)
        else:
            mse_loss = (rgb_l2_loss * soft_mask[:, :, None, :, :]).mean()

        # --- LPIPS Loss (human-only, spatial=True needed for per-pixel weighting) ---
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_human_mask[:, None, ...]).sum() / (flattened_human_mask.sum() + 1e-8)
        else:
            lpips_loss = lpips_loss.mean()

        loss = (
            self.config.loss.mse_loss_weight * mse_loss
            + self.config.loss.lpips_loss_weight * lpips_loss
        )

        # --- Visualization Logging ---
        if hasattr(self, 'global_step') and (self.global_step < 10 or self.global_step % 100 == 0):
            try:
                self.save_visualizations(
                    batch, predicted_color, target_color,
                    human_mask, soft_mask, dataset_human_mask, geom_mask
                )
            except Exception as e:
                print(f"Failed to generate visualization: {e}")

        return loss, mse_loss, lpips_loss

    def save_visualizations(self, batch, predicted_color, target_color, human_mask, soft_mask, dataset_human_mask, geom_mask):
        import torchvision
        import os

        # Only log from rank 0
        if os.getenv("LOCAL_RANK", "0") != "0":
            return

        viz_dir = os.path.join(self.config.save_dir, self.config.name, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        b = 0
        V_tgt = target_color.shape[1]
        img_list = []

        def get_rgb(tensor, v):
            return tensor[b, v].detach().cpu().clamp(0, 1)

        def to_3ch(tensor_mask, v):
            return tensor_mask[b, v].detach().cpu().float().unsqueeze(0).repeat(3, 1, 1)

        # Row 0: Context views (pad with black if fewer than V_tgt)
        num_context = len(batch['context'])
        for v in range(V_tgt):
            if v < num_context:
                img_list.append(batch['context'][v]['original_img'][b].detach().cpu().clamp(0, 1))
            else:
                img_list.append(torch.zeros_like(target_color[b, v]).cpu())

        # Row 1: Target RGB
        for v in range(V_tgt): img_list.append(get_rgb(target_color, v))

        # Row 2: Predicted RGB
        for v in range(V_tgt): img_list.append(get_rgb(predicted_color, v))

        # Row 3: Splatt3R Geom Mask
        for v in range(V_tgt): img_list.append(to_3ch(geom_mask, v))

        # Row 4: Dataset Human Mask
        for v in range(V_tgt): img_list.append(to_3ch(dataset_human_mask, v))

        # Row 5: Intersection Loss Mask (Geom & Dataset)
        for v in range(V_tgt): img_list.append(to_3ch(human_mask, v))

        grid = torchvision.utils.make_grid(img_list, nrow=V_tgt)
        from torchvision.utils import save_image
        step = getattr(self, 'global_step', 0)
        save_path = os.path.join(viz_dir, f"step_{step:06d}.jpg")
        save_image(grid, save_path)
        print(f"Saved visualization to {save_path}")

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None):
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
        }
        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim
        # Enforcing explicit sync_dist=True for Distributed PyTorch Lightning execution over 2+ GPUs
        self.log_dict(values, prog_bar=(prefix != 'val'), sync_dist=True, batch_size=self.config.data.batch_size)

    def configure_optimizers(self):
        # Two param groups with different LRs:
        #   - GCN encoder + prior_attention: train from scratch at full LR
        #   - Gaussian DPT: pretrained weights, 10x lower LR to preserve quality
        gcn_and_attn_params = (
            list(self.gcn_encoder.parameters()) +
            list(self.encoder.downstream_head1.prior_attention.parameters()) +
            list(self.encoder.downstream_head2.prior_attention.parameters())
        )
        dpt_params = (
            list(self.encoder.downstream_head1.gaussian_dpt.dpt.parameters()) +
            list(self.encoder.downstream_head2.gaussian_dpt.dpt.parameters())
        )
        param_groups = [
            {'params': gcn_and_attn_params, 'lr': self.config.opt.lr, 'name': 'gcn_and_attention'},
            {'params': dpt_params, 'lr': self.config.opt.lr * 0.1, 'name': 'gaussian_dpt'},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config.opt.epochs // 2], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

def run_dist_experiment(config):
    L.seed_everything(config.seed, workers=True)
    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)

    # MAST3RAnatomicalRefinement loads the checkpoint internally via
    # MAST3RGaussians.load_from_checkpoint(); config.splatt3r_checkpoint_path
    # must point to a Splatt3R Lightning .ckpt file.
    print('Building MAST3RAnatomicalRefinement...')
    model = MAST3RAnatomicalRefinement(config)

    print('Building Anatomical Datasets')
    dataset_root = getattr(config.data, 'root', '/ssd_scratch/gnrs/mvhumannet++_10demo')

    # Path mappings following DATASET_FORMAT.md
    main_root = os.path.join(dataset_root, 'main')
    depth_root = os.path.join(dataset_root, 'depth')
    anchor_root = os.path.join(dataset_root, 'anchors')
    adj_path = os.path.join(dataset_root, 'smpl_adj_512.npy')
    fps_path = os.path.join(dataset_root, 'fps_indices_512.npy')
    smplx_model_path = getattr(config.data, 'smplx_model_path', 'smplx_models/')

    train_dataset = get_anatomical_dataset(
        main_root=main_root,
        depth_root=depth_root,
        anchor_root=anchor_root,
        adj_path=adj_path,
        fps_indices_path=fps_path,
        smplx_model_path=smplx_model_path,
        resolution=config.data.resolution,
        sequences=getattr(config.data, 'sequences', None),
        num_epochs_per_epoch=getattr(config.data, 'epochs_per_train_epoch', 1),
        num_target_views=getattr(config.data, 'num_target_views', 2)
    )
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=config.data.batch_size,
        num_workers=config.data.num_workers, pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # Setup Loggers
    # -----------------------------------------------------------------------
    loggers = []
    log_cfg = getattr(config, 'loggers', {})
    
    if getattr(log_cfg, 'use_csv_logger', True):
        loggers.append(L.pytorch.loggers.CSVLogger(save_dir=config.save_dir, name=config.name))
        
    if getattr(log_cfg, 'use_wandb', False):
        run_name = getattr(config, 'run_name', None) or getattr(config, 'name', None)
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project='splatt3r-gcn',
            name=run_name,
            save_dir=config.save_dir,
            log_model=False,
        )
        # Force the display name on the active run as some W&B/Lightning
        # combinations can ignore logger name during distributed init.
        if run_name and wandb_logger.experiment is not None:
            wandb_logger.experiment.name = run_name
        loggers.append(wandb_logger)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=config.devices,
        strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
        benchmark=True,
        log_every_n_steps=10,
        max_epochs=config.opt.epochs,
        logger=loggers,
        default_root_dir=config.save_dir,
        accumulate_grad_batches=getattr(config.opt, 'accumulate_grad_batches', 1),
        gradient_clip_val=getattr(config.opt, 'gradient_clip_val', 1.0),
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath='/ssd_scratch/gnrs/gcn_checkpoints',
                monitor='train/loss', save_top_k=3, mode='min',
                filename='anatomical-{epoch:02d}-{train/loss:.4f}'
            ),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
    )
    trainer.fit(model, train_dataloaders=data_loader_train)

    # -----------------------------------------------------------------------
    # Log Final Summary and Best Checkpoint
    # -----------------------------------------------------------------------
    if trainer.is_global_zero:
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else "None"
        best_model_score = checkpoint_callback.best_model_score if checkpoint_callback else "None"
        
        summary_dir = os.path.join(config.save_dir, config.name)
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "training_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("============ TRAINING COMPLETED ============\n")
            f.write(f"Best Checkpoint Path: {best_model_path}\n")
            f.write(f"Best Loss Score: {best_model_score}\n")
            f.write("\nFinal Logged Metrics:\n")
            for k, v in trainer.logged_metrics.items():
                val = v.item() if hasattr(v, 'item') else v
                f.write(f"  {k}: {val}\n")
                
        print(f"\n[Splatt3R-GCN] Saved final training summary and checkpoint paths to: {summary_path}")


if __name__ == "__main__":
    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)
    run_dist_experiment(config)
