import sys
import os
import torch
import numpy as np
from omegaconf import OmegaConf

# Add necessary paths
sys.path.append('.')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

from data.mvhumannet.mvhumannet import get_mvhumannet_dataset, worker_init_fn

def verify_dataset(config_path):
    print(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)
    
    # Check if paths are still placeholders
    if "/path/to/" in config.data.main_root:
        print("ERROR: Please update 'main_root' and 'depth_root' in your config file to absolute paths.")
        return
        
    print(f"Main root: {config.data.main_root}")
    print(f"Depth root: {config.data.depth_root}")
    
    dataset = get_mvhumannet_dataset(
        main_root=config.data.main_root,
        depth_root=config.data.depth_root,
        resolution=tuple(config.data.resolution),
        sequences=config.data.sequences,
        num_epochs_per_epoch=config.data.epochs_per_train_epoch
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Try fetching one item
    print("Fetching first item...")
    batch = dataset[0]
    
    print("\n--- Verification Report ---")
    print(f"Scene: {batch['scene']}")
    print(f"Context views: {len(batch['context'])}")
    print(f"Target views: {len(batch['target'])}")
    
    for i, view in enumerate(batch['context']):
        print(f"\nContext {i}:")
        print(f"  img shape: {view['img'].shape}")
        print(f"  original_img shape: {view['original_img'].shape}")
        print(f"  depthmap shape: {view['depthmap'].shape}")
        print(f"  pts3d (derived from depth) shape: {view['pts3d'].shape}")
        print(f"  valid_mask shape: {view['valid_mask'].shape} (valid ratio: {view['valid_mask'].mean():.2%})")
        print(f"  K:\n{view['camera_intrinsics']}")
        
    for i, view in enumerate(batch['target']):
        print(f"\nTarget {i}:")
        print(f"  original_img shape: {view['original_img'].shape}")
        print(f"  depthmap shape: {view['depthmap'].shape}")
        print(f"  camera_pose:\n{view['camera_pose']}")
        
    print("\nSMPL-X Params:")
    for k, v in batch['smplx'].items():
        print(f"  {k}: {v.shape}")
        
    print("\nSUCCESS: Dataset loading and item fetching works!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_mvhumannet.py <config_path>")
    else:
        verify_dataset(sys.argv[1])
