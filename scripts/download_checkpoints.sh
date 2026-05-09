#!/usr/bin/env bash
# =============================================================================
# download_checkpoints.sh
# Downloads all required model weights for Splatt3R-GCN.
#
# This repo is a branch of: https://github.com/btsmart/splatt3r
# Extended by: https://github.com/ntini773/splatt3r_gcn
#
# Usage:
#   bash scripts/download_checkpoints.sh
#
# What this downloads:
#   1. MASt3R retrieval weights  (official NAVER/MASt3R — CC BY-NC-SA 4.0)
#   2. Base Splatt3R ckpt        (official Splatt3R HuggingFace — CC BY-NC 4.0)
#   3. GCN anatomical ckpt       (our HuggingFace repo — CC BY-NC 4.0)
# =============================================================================

set -e
CKPT_DIR="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"
mkdir -p "$CKPT_DIR"

echo "==> Downloading checkpoints into: $CKPT_DIR"

# ── 1. MASt3R retrieval weights (needed for smart view-pair selection) ───────
MAST3R_RETR="$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
if [ ! -f "$MAST3R_RETR" ]; then
  echo "Downloading MASt3R retrieval weights..."
  wget -q --show-progress -O "$MAST3R_RETR" \
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
else
  echo "  [skip] MASt3R retrieval weights already present"
fi

# ── 2. Base Splatt3R checkpoint ────────────────────────────────────────────────
SPLATT3R_CKPT="/ssd_scratch/$USER/checkpoints/epoch=19-step=1200.ckpt"
mkdir -p "$(dirname "$SPLATT3R_CKPT")"

if [ ! -f "$SPLATT3R_CKPT" ]; then
  echo "Downloading base Splatt3R checkpoint to $SPLATT3R_CKPT ..."
  wget -q --show-progress -O "$SPLATT3R_CKPT" \
    "https://huggingface.co/brandonsmart/splatt3r_v1.0/resolve/main/epoch%3D19-step%3D1200.ckpt"
else
  echo "  [skip] Splatt3R base checkpoint already present"
fi

# ── 3. GCN anatomical refinement checkpoint ────────────────────────────────
GCN_CKPT="/ssd_scratch/$USER/gcn_checkpoints/anatomical-epoch=21-train/loss=0.3257.ckpt"
mkdir -p "$(dirname "$GCN_CKPT")"

if [ ! -f "$GCN_CKPT" ]; then
  echo "Downloading GCN anatomical refinement checkpoint (private repo — requires HF token)..."
  echo "  Run: huggingface-cli login   if you haven't already."
  python3 - <<EOF
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id="Nitin773/splatt3r-gcn",
    filename="anatomical-epoch21.ckpt",
    repo_type="model",
    local_dir="/tmp",
)
shutil.move(path, "$GCN_CKPT")
print(f"  Saved to: $GCN_CKPT")
EOF
else
  echo "  [skip] GCN checkpoint already present"
fi

# ── SMPL-X (requires registration) ─────────────────────────────────────────────
echo ""
echo "NOTE: SMPL-X model weights require registration (not auto-downloaded)."
echo "  1. Register at https://smpl-x.is.tue.mpg.de"
echo "  2. Download 'SMPL-X v1.1' and place at: checkpoints/smplx/models/"
echo ""

# ── 4. PEAR assets (SMPL/SMPLX/MANO body models — private HF backup) ─────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PEAR_ASSETS_DIR="$REPO_ROOT/third_party/PEAR/assets"
if [ ! -d "$PEAR_ASSETS_DIR/SMPL" ] && [ ! -d "$PEAR_ASSETS_DIR/SMPLX" ]; then
  echo "Downloading PEAR assets from private HF repo (requires HF token)..."
  mkdir -p "$PEAR_ASSETS_DIR"
  python3 - <<EOF
from huggingface_hub import hf_hub_download
import zipfile, os

print("  Downloading zipped PEAR assets...")
zip_path = hf_hub_download(
    repo_id="Nitin773/splatt3r-gcn-assets",
    filename="pear_assets.zip",
    repo_type="model",
    local_dir="/tmp",
)

print(f"  Unzipping to $PEAR_ASSETS_DIR ...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("$PEAR_ASSETS_DIR")

os.remove(zip_path)
print(f"  PEAR assets restored to: $PEAR_ASSETS_DIR")
EOF
else
  echo "  [skip] PEAR assets already present at $PEAR_ASSETS_DIR"
fi

echo ""
echo "==> All downloads complete."
echo "  Next: run  bash scripts/download_dataset.sh  to get processed MVHumanNet++ dataset."
