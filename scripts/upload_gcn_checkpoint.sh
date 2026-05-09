#!/usr/bin/env bash
# =============================================================================
# upload_gcn_checkpoint.sh
# Uploads your trained GCN anatomical refinement checkpoint to HuggingFace.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login   (paste your HF write token)
#
# Usage:
#   bash scripts/upload_gcn_checkpoint.sh /path/to/checkpoint.ckpt
# =============================================================================

set -e

CKPT_PATH="${1:-/ssd_scratch/gnrs/gcn_checkpoints/anatomical-epoch=21-train/loss=0.3257.ckpt}"
HF_REPO="gnrs/splatt3r-gcn"       # change to your HuggingFace username/repo-name
HF_FILENAME="anatomical-epoch21.ckpt"

if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found at $CKPT_PATH"
  exit 1
fi

echo "Uploading $CKPT_PATH → hf.co/$HF_REPO/$HF_FILENAME ..."

python3 - <<EOF
from huggingface_hub import HfApi, create_repo
api = HfApi()

# Create repo if it doesn't exist (private=False for public access)
try:
    create_repo("$HF_REPO", repo_type="model", private=False, exist_ok=True)
    print("Repo ready: https://huggingface.co/$HF_REPO")
except Exception as e:
    print(f"Repo already exists or error: {e}")

# Upload checkpoint
api.upload_file(
    path_or_fileobj="$CKPT_PATH",
    path_in_repo="$HF_FILENAME",
    repo_id="$HF_REPO",
    repo_type="model",
    commit_message="Add GCN anatomical refinement checkpoint (epoch 21)",
)
print("Upload complete!")
print("Download URL: https://huggingface.co/$HF_REPO/resolve/main/$HF_FILENAME")
EOF
