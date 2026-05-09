#!/usr/bin/env bash
# =============================================================================
# upload_all_to_hf.sh
# One-shot script: uploads GCN checkpoint + full dataset + PEAR assets to HF.
#
# Prerequisites:
#   hf auth login   (use your HF Write token)
#
# Usage:
#   bash scripts/upload_all_to_hf.sh
# =============================================================================

set -e

HF_USER="Nitin773"
CKPT_REPO="$HF_USER/splatt3r-gcn"
DATA_REPO="$HF_USER/splatt3r-gcn-data"
ASSETS_REPO="$HF_USER/splatt3r-gcn-assets"

GCN_CKPT="/ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=21-train/loss=0.3257.ckpt"
DATA_ROOT="/ssd_scratch/gnrs/mvhumannet++_10demo"
DATA_ZIP="${DATA_ROOT}_processed.zip"
PEAR_ASSETS="$(cd "$(dirname "$0")/.." && pwd)/third_party/PEAR/assets"
PEAR_ZIP="$(cd "$(dirname "$0")/.." && pwd)/third_party/PEAR/pear_assets.zip"

echo "=== Uploading to HuggingFace as: $HF_USER ==="

# ── 1. Upload GCN checkpoint ──────────────────────────────────────────────────
echo ""
echo "[1/3] Uploading GCN checkpoint → hf.co/$CKPT_REPO ..."
python3 - <<EOF
from huggingface_hub import HfApi, create_repo
api = HfApi()
create_repo("$CKPT_REPO", repo_type="model", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj="$GCN_CKPT",
    path_in_repo="anatomical-epoch21.ckpt",
    repo_id="$CKPT_REPO",
    repo_type="model",
    commit_message="Add GCN anatomical refinement checkpoint (epoch 21, loss=0.3257)",
)
print(f"  Model uploaded: https://huggingface.co/$CKPT_REPO")
EOF

# ── 2. Upload full processed dataset (private — personal backup only) ─────────
echo ""
echo "[2/3] Uploading full processed dataset → hf.co/datasets/$DATA_REPO ..."
echo "      (Private repo — personal backup only, do NOT make public)"
echo "  Zipping dataset to $DATA_ZIP (this may take a while)..."
(cd "$DATA_ROOT" && zip -rq "$DATA_ZIP" .)

python3 - <<EOF
from huggingface_hub import HfApi, create_repo
api = HfApi()
create_repo("$DATA_REPO", repo_type="dataset", private=True, exist_ok=True)
print("  Uploading zipped dataset...")
api.upload_file(
    path_or_fileobj="$DATA_ZIP",
    path_in_repo="mvhumannet_processed.zip",
    repo_id="$DATA_REPO",
    repo_type="dataset",
    commit_message="Full processed MVHumanNet++ dataset (zipped backup)",
)
print(f"  Dataset uploaded: https://huggingface.co/datasets/$DATA_REPO")
EOF

# ── 3. Upload PEAR assets (SMPL/SMPLX/MANO — private, licensed models) ────────
echo ""
echo "[3/3] Uploading PEAR assets → hf.co/$ASSETS_REPO ..."
echo "      Contents: SMPL, SMPLX, MANO, SMPLX2SMPL body model files"
echo "      (Private — licensed body models, do NOT make public)"
echo "  Zipping PEAR assets to $PEAR_ZIP..."
(cd "$PEAR_ASSETS" && zip -rq "$PEAR_ZIP" .)

python3 - <<EOF
from huggingface_hub import HfApi, create_repo
api = HfApi()
create_repo("$ASSETS_REPO", repo_type="model", private=True, exist_ok=True)
print("  Uploading zipped PEAR assets...")
api.upload_file(
    path_or_fileobj="$PEAR_ZIP",
    path_in_repo="pear_assets.zip",
    repo_id="$ASSETS_REPO",
    repo_type="model",
    commit_message="PEAR assets: SMPL/SMPLX/MANO body models (zipped backup)",
)
print(f"  Assets uploaded: https://huggingface.co/$ASSETS_REPO")
EOF


echo ""
echo "=== Upload complete! ==="
echo "  Checkpoint : https://huggingface.co/$CKPT_REPO"
echo "  Dataset    : https://huggingface.co/datasets/$DATA_REPO"
echo "  PEAR assets: https://huggingface.co/$ASSETS_REPO"
