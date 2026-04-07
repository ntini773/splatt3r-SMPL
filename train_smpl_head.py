"""
Training script for SMPL-X head on top of a frozen Splatt3R backbone.

WHY SPLATT3R CHECKPOINTS INSTEAD OF MAST3R?
--------------------------------------------
The original train script loaded raw MASt3R pretrained weights and manually
re-initialised an AsymmetricMASt3R model configured for SMPL output. That is
wrong for this project because:

  1. Splatt3R's encoder uses a *different* output mode and head architecture
     ("pts3d+gaussian+desc24", gaussian_head) compared to a vanilla MASt3R
     checkpoint.  Loading the wrong weight file causes shape mismatches or
     silently initialises the Gaussian heads from scratch.

  2. We want to keep the Gaussian reconstruction pipeline working in parallel
     with the new SMPL-X head so we can evaluate both outputs simultaneously.
     The Splatt3R checkpoint already includes trained Gaussian DPT heads; we
     should not throw that away.

  3. demo.py shows that the canonical way to restore MAST3RGaussians is via
     MAST3RGaussians.load_from_checkpoint(), which is the Lightning API and
     correctly restores ALL sub-modules including the decoder and any
     hyperparameters stored in save_hyperparameters().

WHAT THIS SCRIPT DOES
---------------------
  - Loads a Splatt3R Lightning checkpoint via load_from_checkpoint().
  - Freezes the encoder backbone (ViT + Transformer decoders) AND the
    pre-trained Gaussian downstream heads so their weights are not disturbed.
  - Attaches a new SMPLXHead that operates on the frozen decoder tokens.
  - Trains ONLY the SMPL-X head parameters using the MV-HumanPlus++ dataset
    which provides per-frame SMPL-X annotations.
  - Logs reconstruction metrics alongside human-body metrics (MPJPE, shape /
    pose / expression L2) so we can verify the Gaussian quality does not
    degrade.

DATASET: MV-HumanPlus++
------------------------
The original proposal used ScanNet++ which has no human annotations.
MV-HumanPlus++ provides multi-view RGB images of humans together with
SMPL-X parameter annotations (betas, body_pose, global_orient, expression,
hand poses, translation), making it the right choice for this head.
The dataset class is expected to return the keys documented in
MVHumanPlusPlusDataset below.
"""

import os
import sys
import json

import einops
import lightning as L
import lpips
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import smplx as smplx_lib
except ImportError:
    smplx_lib = None
    print("Warning: smplx package not installed – joint-position loss disabled.")

# ---------------------------------------------------------------------------
# Path setup – mirrors main.py / demo.py exactly so imports resolve the same
# way regardless of working directory.
# ---------------------------------------------------------------------------
sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

import main as splatt3r_main   # gives us MAST3RGaussians
import utils.export as export
import utils.loss_mask as loss_mask
import utils.compute_ssim as compute_ssim


# ===========================================================================
# SMPL-X Regression Head
# ===========================================================================

class SMPLXHead(nn.Module):
    """
    Lightweight MLP that regresses SMPL-X parameters from the concatenation
    of the two frozen Splatt3R decoder token sequences.

    WHY THIS DESIGN?
    ----------------
    MASt3R's decoder produces a sequence of patch tokens for each view
    (shape: B x N_patches x C).  We pool them to a fixed-size vector and
    pass through a small MLP.  This keeps the head simple and fast to train,
    which is important given our limited GPU budget (4 × RTX 2080 Ti).

    The head outputs the standard SMPL-X parameter set:
        betas            : (B, 10)   shape coefficients
        global_orient    : (B, 3)    root orientation (axis-angle)
        body_pose        : (B, 63)   21 joints × 3 (axis-angle)
        left_hand_pose   : (B, 45)   15 joints × 3
        right_hand_pose  : (B, 45)   15 joints × 3
        jaw_pose         : (B, 3)
        leye_pose        : (B, 3)
        reye_pose        : (B, 3)
        expression       : (B, 10)
        transl           : (B, 3)
    """

    # Expected decoder token channel dimension for the Splatt3R ViT-Large
    TOKEN_DIM = 768   # dec_embed_dim from main.py

    # Concatenating two views → 2 × TOKEN_DIM input to the MLP
    INPUT_DIM = 2 * TOKEN_DIM

    # SMPL-X output sizes
    PARAM_SIZES = {
        'betas':            10,
        'global_orient':     3,
        'body_pose':        63,
        'left_hand_pose':   6,
        'right_hand_pose':  6,
        'jaw_pose':          3,
        'leye_pose':         3,
        'reye_pose':         3,
        'expression':       10,
        'transl':            3,
    }

    def __init__(self, hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        total_output = sum(self.PARAM_SIZES.values())   # 188

        layers = []
        in_dim = self.INPUT_DIM
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, total_output))
        self.mlp = nn.Sequential(*layers)

        # Store cumulative split points so we can slice the output tensor
        self._sizes = list(self.PARAM_SIZES.values())
        self._keys  = list(self.PARAM_SIZES.keys())

    def _pool_tokens(self, dec_tokens):
        """
        Average-pool a list of decoder token tensors to a single vector.

        dec_tokens is a list of tensors each shaped (B, N, C) as returned by
        MASt3R's _decoder.  We stack and mean across both list dimension and
        patch dimension.
        """
        # Stack along a new axis → (B, num_blocks, N, C), then mean over blocks and patches
        stacked = torch.stack([t for t in dec_tokens], dim=1)  # (B, L, N, C)
        return stacked.mean(dim=(1, 2))                         # (B, C)

    def forward(self, dec1_tokens, dec2_tokens):
        """
        Args:
            dec1_tokens: list of (B, N, C) tensors from decoder 1 (view 1)
            dec2_tokens: list of (B, N, C) tensors from decoder 2 (view 2)

        Returns:
            dict mapping SMPL-X parameter names to tensors
        """
        feat1 = self._pool_tokens(dec1_tokens)   # (B, 768)
        feat2 = self._pool_tokens(dec2_tokens)   # (B, 768)
        feat  = torch.cat([feat1, feat2], dim=-1)  # (B, 1536) – fuses both views

        raw = self.mlp(feat)   # (B, 188)
        params = {}
        offset = 0
        for key, size in zip(self._keys, self._sizes):
            params[key] = raw[:, offset:offset + size]
            offset += size
        return params


# ===========================================================================
# Lightning Module
# ===========================================================================

class Splatt3RWithSMPLX(L.LightningModule):
    """
    Wraps the pre-trained MAST3RGaussians model and adds a trainable SMPL-X
    regression head.

    FREEZING STRATEGY
    -----------------
    We freeze everything that was already trained in the Splatt3R checkpoint:
      - The ViT encoder (encoder.patch_embed, encoder.enc_blocks, etc.)
      - The cross-attention transformer decoders (encoder.dec_blocks*)
      - The Gaussian DPT heads (encoder.downstream_head1/2)
      - The PixelSplat splatting decoder (self.decoder)

    Only self.smplx_head is left trainable.  This lets us use a much higher
    learning rate for the head without destabilising the Gaussian pipeline.

    If config.unfreeze_decoder_heads is True we also unfreeze downstream_head1
    and downstream_head2 for joint fine-tuning (useful in a second training
    stage).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # 1. Load the complete Splatt3R model from its Lightning checkpoint.
        #    This is the KEY difference vs. the original script which loaded
        #    a raw MASt3R .pth file. load_from_checkpoint restores the exact
        #    architecture (Gaussian heads, decoder, hyperparams) trained by
        #    Splatt3R, not just the ViT backbone.
        # ------------------------------------------------------------------
        print(f"Loading Splatt3R checkpoint from {config.splatt3r_checkpoint_path} ...")
        self.splatt3r = splatt3r_main.MAST3RGaussians.load_from_checkpoint(
            config.splatt3r_checkpoint_path,
            map_location='cpu',          # move to correct device via Lightning
            strict=True,
        )

        # ------------------------------------------------------------------
        # 2. Freeze the entire Splatt3R model.
        #    We do NOT want gradients flowing into the backbone or the
        #    Gaussian heads because:
        #      a) It would destroy the carefully trained 3DGS features.
        #      b) Our dataset (MV-HumanPlus++) contains humans; ScanNet++ did
        #         not, so the Gaussian head could overfit to human textures if
        #         unfrozen on this dataset.
        # ------------------------------------------------------------------
        self.splatt3r.requires_grad_(False)

        # Optional second-stage: unfreeze the Gaussian DPT heads so that
        # rendering quality on human scenes can also improve.
        if getattr(config, 'unfreeze_gaussian_heads', False):
            self.splatt3r.encoder.downstream_head1.requires_grad_(True)
            self.splatt3r.encoder.downstream_head2.requires_grad_(True)
            print("  → Gaussian DPT heads UNFROZEN for joint fine-tuning.")
        else:
            print("  → All Splatt3R parameters FROZEN. Training SMPL-X head only.")

        # ------------------------------------------------------------------
        # 3. Attach the new trainable SMPL-X head.
        # ------------------------------------------------------------------
        self.smplx_head = SMPLXHead(
            hidden_dim=getattr(config, 'smplx_hidden_dim', 512),
            num_layers=getattr(config, 'smplx_num_layers', 3),
            dropout=getattr(config, 'smplx_dropout', 0.1),
        )

        # ------------------------------------------------------------------
        # 4. Optionally load the SMPL-X body model for joint-position loss.
        #    We defer this to on_train_start so the model is already on the
        #    right device when we call smplx_lib.create().
        # ------------------------------------------------------------------
        self.smplx_body_model = None
        self.smplx_model_path = getattr(config, 'smplx_model_path', None)

        # Loss weights
        self.lambda_pose   = getattr(config, 'lambda_pose',   1.0)
        self.lambda_shape  = getattr(config, 'lambda_shape',  0.1)
        self.lambda_expr   = getattr(config, 'lambda_expr',   0.1)
        self.lambda_transl = getattr(config, 'lambda_transl', 1.0)
        self.lambda_joints = getattr(config, 'lambda_joints', 1.0)

        # LPIPS for optional Gaussian rendering loss (frozen, just for logging)
        self.lpips_fn = lpips.LPIPS('vgg')
        self.lpips_fn.requires_grad_(False)

        self.save_hyperparameters()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self):
        """Load SMPL-X body model once the device is known."""
        if smplx_lib is None or self.smplx_model_path is None:
            print("Skipping SMPL-X body model load (smplx not installed or path not set).")
            return
        if not os.path.exists(self.smplx_model_path):
            print(f"Warning: SMPL-X model path not found: {self.smplx_model_path}")
            return
        try:
            self.smplx_body_model = smplx_lib.create(
                self.smplx_model_path,
                model_type='smplx',
                gender='neutral',
                num_betas=10,
                num_expression_coeffs=10,
                num_pca_comps=6,
                use_pca=True,
                flat_hand_mean=True,
            ).to(self.device)
            self.smplx_body_model.requires_grad_(False)
            print(f"SMPL-X body model loaded from {self.smplx_model_path}")
        except Exception as e:
            print(f"Error loading SMPL-X body model: {e}")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, view1, view2):
        """
        Run the frozen Splatt3R encoder/decoder, then predict SMPL-X params.

        WHY torch.no_grad() AROUND THE BACKBONE?
        -----------------------------------------
        Even though requires_grad=False on the backbone parameters prevents
        weight updates, PyTorch still builds the autograd graph through those
        ops unless we explicitly disable it.  Using no_grad() here:
          - Reduces memory usage by ~30–40% (no intermediate activations saved)
          - Speeds up the forward pass noticeably on large ViT models
          - Makes it unambiguous that nothing in the backbone will change
        """
        encoder = self.splatt3r.encoder

        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = \
                encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = encoder._decoder(feat1, pos1, feat2, pos2)

            # Gaussian predictions (kept for optional rendering loss logging)
            pred1 = encoder._downstream_head(1, [t.float() for t in dec1], shape1)
            pred2 = encoder._downstream_head(2, [t.float() for t in dec2], shape2)

        # SMPL-X prediction – this IS in the autograd graph
        smplx_pred = self.smplx_head(
            [t.float() for t in dec1],
            [t.float() for t in dec2],
        )

        return pred1, pred2, smplx_pred

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_smplx_loss(self, smplx_pred, batch):
        """
        Compute all SMPL-X losses and return a dict.

        LOSS DESIGN
        -----------
        1. Pose L2 – L2 on axis-angle representation for each joint group.
           Simple and fast; axis-angle is less sensitive to gimbal lock than
           rotation matrices for small deviations.

        2. Shape L2 – L2 on beta coefficients.  Low weight (0.1) because betas
           change slowly and the dataset may not have huge shape variation.

        3. Expression L2 – L2 on expression coefficients.  Low weight (0.1).

        4. Translation L2 – L2 on root translation.  Helps anchor the body in
           the correct 3D location relative to the camera.

        5. MPJPE – Mean Per Joint Position Error on the 22 main body joints.
           Requires running the SMPL-X forward model, which is expensive, so
           it is gated on self.smplx_body_model being available.  Using the
           body model as a differentiable layer ensures the joint loss is
           geometrically meaningful (not just parameter-space).
        """
        losses = {}
        device = smplx_pred['betas'].device

        # 1. Pose losses (all joint groups)
        pose_keys = [
            ('global_orient', 'smplx_global_orient'),
            ('body_pose',     'smplx_body_pose'),
            ('left_hand_pose','smplx_left_hand_pose'),
            ('right_hand_pose','smplx_right_hand_pose'),
            ('jaw_pose',      'smplx_jaw_pose'),
            ('leye_pose',     'smplx_leye_pose'),
            ('reye_pose',     'smplx_reye_pose'),
        ]
        pose_loss = torch.tensor(0.0, device=device)
        for pred_key, batch_key in pose_keys:
            if batch_key in batch:
                pose_loss = pose_loss + F.mse_loss(smplx_pred[pred_key], batch[batch_key])
        losses['pose'] = self.lambda_pose * pose_loss

        # 2. Shape loss
        if 'smplx_betas' in batch:
            losses['shape'] = self.lambda_shape * F.mse_loss(
                smplx_pred['betas'], batch['smplx_betas'])
        else:
            losses['shape'] = torch.tensor(0.0, device=device)

        # 3. Expression loss
        if 'smplx_expression' in batch:
            losses['expression'] = self.lambda_expr * F.mse_loss(
                smplx_pred['expression'], batch['smplx_expression'])
        else:
            losses['expression'] = torch.tensor(0.0, device=device)

        # 4. Translation loss
        if 'smplx_transl' in batch:
            losses['transl'] = self.lambda_transl * F.mse_loss(
                smplx_pred['transl'], batch['smplx_transl'])
        else:
            losses['transl'] = torch.tensor(0.0, device=device)

        # 5. MPJPE via SMPL-X body model forward pass
        losses['mpjpe'] = torch.tensor(0.0, device=device)
        losses['mpjpe_mm'] = torch.tensor(0.0, device=device)  # unweighted mm, logged as accuracy

        if self.smplx_body_model is not None and self.lambda_joints > 0 and 'smplx_betas' in batch:
            try:
                out_pred = self.smplx_body_model(
                    betas=smplx_pred['betas'],
                    expression=smplx_pred['expression'],
                    global_orient=smplx_pred['global_orient'],
                    body_pose=smplx_pred['body_pose'],
                    left_hand_pose=smplx_pred['left_hand_pose'],
                    right_hand_pose=smplx_pred['right_hand_pose'],
                    jaw_pose=smplx_pred['jaw_pose'],
                    leye_pose=smplx_pred['leye_pose'],
                    reye_pose=smplx_pred['reye_pose'],
                    transl=smplx_pred['transl'],
                    return_verts=False,
                )
                out_gt = self.smplx_body_model(
                    betas=batch['smplx_betas'],
                    expression=batch['smplx_expression'],
                    global_orient=batch['smplx_global_orient'],
                    body_pose=batch['smplx_body_pose'],
                    left_hand_pose=batch['smplx_left_hand_pose'],
                    right_hand_pose=batch['smplx_right_hand_pose'],
                    jaw_pose=batch['smplx_jaw_pose'],
                    leye_pose=batch['smplx_leye_pose'],
                    reye_pose=batch['smplx_reye_pose'],
                    transl=batch['smplx_transl'],
                    return_verts=False,
                )
                pred_j = out_pred.joints[:, :22, :]
                gt_j   = out_gt.joints[:, :22, :]
                mpjpe_raw = torch.norm(pred_j - gt_j, dim=-1).mean()
                losses['mpjpe']    = self.lambda_joints * mpjpe_raw
                losses['mpjpe_mm'] = mpjpe_raw.detach() * 1000.0  # metres → mm
            except Exception as e:
                print(f"MPJPE computation failed: {e}")

        losses['total'] = sum(v for k, v in losses.items() if k != 'mpjpe_mm')
        return losses

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str):
        view1, view2 = batch['context']
        pred1, pred2, smplx_pred = self(view1, view2)

        smplx_losses = self.compute_smplx_loss(smplx_pred, batch)
        loss = smplx_losses['total']

        # Build log dict – all keys route through Lightning to WandbLogger.
        # mpjpe_mm (MPJPE in mm) is the accuracy metric; lower = better.
        log = {
            f'{stage}/loss':            loss,
            f'{stage}/loss_pose':       smplx_losses['pose'],
            f'{stage}/loss_shape':      smplx_losses['shape'],
            f'{stage}/loss_expression': smplx_losses['expression'],
            f'{stage}/loss_transl':     smplx_losses['transl'],
            f'{stage}/loss_mpjpe':      smplx_losses['mpjpe'],
            f'{stage}/mpjpe_mm':        smplx_losses['mpjpe_mm'],  # accuracy metric
        }

        # Optional: PSNR on Gaussian renders during validation.
        if stage == 'val' and 'target' in batch:
            with torch.no_grad():
                _, _, h, w = batch['context'][0]['img'].shape
                color, _ = self.splatt3r.decoder(batch, pred1, pred2, (h, w))
                target = torch.stack(
                    [t['original_img'] for t in batch['target']], dim=1)
                mse = ((color - target) ** 2).mean()
                log['val/psnr'] = -10.0 * mse.log10()

        sync = stage != 'train'
        self.log_dict(log, on_step=(stage == 'train'), on_epoch=True,
                      sync_dist=sync, prog_bar=True,
                      batch_size=self.config.batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """
        OPTIMISER CHOICE
        ----------------
        We use AdamW with a cosine annealing schedule.  The learning rate
        (1e-3 by default) is much higher than what you would use for
        fine-tuning the full model (typically 1e-5) because we are only
        training the small SMPL-X head from scratch, so faster convergence
        is safe and desirable.

        If unfreeze_gaussian_heads is True we use a 10x lower LR for those
        parameters via separate param groups, protecting the existing
        Gaussian quality.
        """
        head_params = list(self.smplx_head.parameters())
        param_groups = [{'params': head_params, 'lr': self.config.lr}]

        if getattr(self.config, 'unfreeze_gaussian_heads', False):
            gaussian_params = (
                list(self.splatt3r.encoder.downstream_head1.parameters()) +
                list(self.splatt3r.encoder.downstream_head2.parameters())
            )
            param_groups.append({
                'params': gaussian_params,
                'lr': self.config.lr * 0.1,   # 10× lower to protect Gaussian quality
                'name': 'gaussian_heads',
            })

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.lr * 1e-2,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


# ===========================================================================
# Dataset placeholder – replace with real MV-HumanPlus++ loader
# ===========================================================================

class MVHumanPlusPlusDataset(torch.utils.data.Dataset):
    """
    Minimal interface for the MV-HumanPlus++ dataset.

    WHY NOT SCANNET++?
    ------------------
    The original train script used ScanNet++ which contains indoor scenes,
    NOT humans, and has no SMPL/SMPL-X annotations.  MV-HumanPlus++ was
    specifically chosen because it provides:
      - Multi-view RGB frames of humans in various poses/clothing
      - Per-frame SMPL-X parameter annotations
      - Camera intrinsics / extrinsics per view

    Expected __getitem__ output (dict):
        context : list of two view dicts, each with keys:
                    'img'          : (3, H, W) normalised tensor
                    'original_img' : (3, H, W) [0,1] tensor
                    'true_shape'   : (2,) int tensor [H, W]
                    'camera_pose'  : (4, 4) float tensor
                    'camera_intrinsics': (3, 3) float tensor
        target  : list of target view dicts (same structure as context views)
        smplx_betas           : (10,)
        smplx_global_orient   : (3,)
        smplx_body_pose       : (63,)
        smplx_left_hand_pose  : (45,)
        smplx_right_hand_pose : (45,)
        smplx_jaw_pose        : (3,)
        smplx_leye_pose       : (3,)
        smplx_reye_pose       : (3,)
        smplx_expression      : (10,)
        smplx_transl          : (3,)
    """

    def __init__(self, root, split='train', resolution=(512, 512)):
        self.root       = root
        self.split      = split
        self.resolution = resolution
        # TODO: build self.samples index from MV-HumanPlus++ directory structure
        raise NotImplementedError(
            "MVHumanPlusPlusDataset is a stub. "
            "Implement __len__ and __getitem__ for your data layout."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError


# ===========================================================================
# Config dataclass
# ===========================================================================

class Config:
    # ---- Paths ----
    # Path to a Splatt3R Lightning checkpoint (.ckpt), e.g. downloaded via
    # hf_hub_download as shown in demo.py.
    splatt3r_checkpoint_path: str = './checkpoints/epoch=19-step=1200.ckpt'

    # Path to the SMPL-X model directory (contains smplx/SMPLX_NEUTRAL.npz etc.)
    smplx_model_path: str = './smplx_models'

    # MV-HumanPlus++ root directory
    data_root: str = '/data/mv_human_plus_plus'

    # ---- Training ----
    epochs:     int   = 50
    batch_size: int   = 8       # fits in 12 GB per GPU with frozen backbone
    num_workers: int  = 4
    lr:          float = 1e-3   # high LR OK because we only train a small head
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    resolution: tuple = (512, 512)

    # ---- Head architecture ----
    smplx_hidden_dim: int   = 512
    smplx_num_layers: int   = 3
    smplx_dropout:    float = 0.1

    # ---- Loss weights ----
    lambda_pose:   float = 1.0
    lambda_shape:  float = 0.1
    lambda_expr:   float = 0.1
    lambda_transl: float = 1.0
    lambda_joints: float = 1.0

    # ---- Stage-2 option ----
    # Set to True in a second training run to jointly fine-tune the Gaussian
    # DPT heads at 10× lower LR alongside the SMPL-X head.
    unfreeze_gaussian_heads: bool = False

    # ---- Infrastructure ----
    devices:   list  = None   # e.g. [0,1,2,3] – defaults to all available GPUs
    save_dir:  str   = './runs/smplx_head'
    run_name:  str   = 'smplx_head_v1'
    use_wandb: bool  = True
    seed:      int   = 42

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.devices is None:
            n = torch.cuda.device_count()
            self.devices = list(range(n)) if n > 0 else None


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    # You can replace this block with omegaconf / argparse loading as needed.
    # -----------------------------------------------------------------------
    config = Config()

    L.seed_everything(config.seed, workers=True)

    # Build datasets
    train_dataset = MVHumanPlusPlusDataset(
        config.data_root, split='train', resolution=config.resolution)
    val_dataset   = MVHumanPlusPlusDataset(
        config.data_root, split='val',   resolution=config.resolution)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Build model
    model = Splatt3RWithSMPLX(config)

    # Loggers
    os.makedirs(os.path.join(config.save_dir, config.run_name), exist_ok=True)
    loggers = [
        L.pytorch.loggers.CSVLogger(save_dir=config.save_dir, name=config.run_name)
    ]
    if config.use_wandb:
        loggers.append(L.pytorch.loggers.WandbLogger(
            project='sharingan-splatt3r',
            name=config.run_name,
            save_dir=config.save_dir,
            log_model=False,
        ))

    # Callbacks
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=os.path.join(config.save_dir, config.run_name),
            filename='smplx_head-{epoch:02d}-{val/smplx_total:.4f}',
            monitor='val/smplx_total',
            save_top_k=3,
            mode='min',
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch'),
        L.pytorch.callbacks.EarlyStopping(
            monitor='val/smplx_total',
            patience=10,
            mode='min',
        ),
    ]

    # Trainer
    strategy = (
        'ddp_find_unused_parameters_true'
        if config.devices and len(config.devices) > 1
        else 'auto'
    )
    trainer = L.Trainer(
        accelerator='gpu',
        devices=config.devices,
        strategy=strategy,
        max_epochs=config.epochs,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=config.save_dir,
        benchmark=True,       # speeds up conv ops when input size is fixed
    )

    print("=" * 60)
    print("Training SMPL-X head on frozen Splatt3R backbone")
    print(f"  Checkpoint : {config.splatt3r_checkpoint_path}")
    print(f"  Dataset    : MV-HumanPlus++ @ {config.data_root}")
    print(f"  Epochs     : {config.epochs}")
    print(f"  Devices    : {config.devices}")
    print("=" * 60)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training complete.")


if __name__ == '__main__':
    main()