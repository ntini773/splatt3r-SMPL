#!/usr/bin/env bash
# =============================================================================
# download_dataset.sh
# Downloads the processed MVHumanNet++ anchor data from our HuggingFace repo.
#
# What is distributed here (non-image, copyright-safe):
#   - smpl_adj_512.npy         : SMPL-X body graph adjacency (topology only)
#   - fps_indices_512.npy      : FPS vertex indices into SMPL-X mesh (topology only)
#   - anchors/<seq_id>/*.npy   : Per-frame SMPL-X vertex positions (pose params,
#                                NOT raw images — requires MVHumanNet++ license)
#
# IMPORTANT: By downloading the anchor files you acknowledge that:
#   1. You have obtained (or will obtain) access to MVHumanNet++ from the
#      official source: https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet
#   2. These files are derived from that dataset and are shared under the
#      same terms (research / non-commercial use only).
#
# Usage:
#   bash scripts/download_dataset.sh [output_dir]
#   Default output_dir: /ssd_scratch/$USER/mvhumannet_processed
# =============================================================================

set -e

OUT_DIR="${1:-/ssd_scratch/$USER/mvhumannet++_10demo}"
HF_REPO="Nitin773/splatt3r-gcn-data"   # our HuggingFace dataset repo

echo "==> Downloading processed MVHumanNet++ anchors to: $OUT_DIR"
echo "    (Private repo — requires HF token. Run: huggingface-cli login)"
mkdir -p "$OUT_DIR"

# Requires: pip install huggingface_hub  (already in environment)
python3 - <<EOF
from huggingface_hub import hf_hub_download
import zipfile, os

print("  Downloading zipped dataset (this may take a while)...")
zip_path = hf_hub_download(
    repo_id="$HF_REPO",
    filename="mvhumannet_processed.zip",
    repo_type="dataset",
    local_dir="/tmp",
)


print(f"  Unzipping to $OUT_DIR ...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("$OUT_DIR")

os.remove(zip_path)
print("  Done.")
EOF

echo ""
echo "==> Done. Update your config paths:"
echo "  dataset_root:     $OUT_DIR"
echo "  adj_path:         $OUT_DIR/smpl_adj_512.npy"
echo "  fps_indices_path: $OUT_DIR/fps_indices_512.npy"
echo ""
echo "NOTE: Raw images are NOT included. Download them from:"
echo "  https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet"
echo "  and place under $OUT_DIR/main/<seq_id>/images/"
