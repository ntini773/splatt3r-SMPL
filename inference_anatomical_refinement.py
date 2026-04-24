import os
import sys
import torch
import einops
import matplotlib.pyplot as plt

sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
from train_anatomical_refinement import MAST3RAnatomicalRefinement
from src.anatomical_prior.anatomical_dataset import get_anatomical_dataset
import workspace


@torch.no_grad()
def infer_zero_shot(config, model_checkpoint):
    """
    Minimal zero-shot inference utilizing the offline anatomical meshes 
    appended onto standard RGB inputs. 
    """
    # 1. Load Architecture
    print(f"Loading checkpoint {model_checkpoint}...")
    # First it natively loads the base Splatt3R checkpoint during __init__
    model = MAST3RAnatomicalRefinement(config)
    
    # Overwrite the base weights with our newly fine-tuned GCN/Attention weights!
    if os.path.exists(model_checkpoint):
        ckpt = torch.load(model_checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)
    else:
        print(f"Warning: Fine-tuned checkpoint {model_checkpoint} not found! Running with bare base weights.")

    model.cuda()
    model.eval()

    # 2. Setup Validation Dataloader using correct path mappings
    dataset_root = getattr(config.data, 'root', '/ssd_scratch/gnrs/mvhumannet++_10demo')
    
    dataset = get_anatomical_dataset(
        main_root=os.path.join(dataset_root, 'main'),
        depth_root=os.path.join(dataset_root, 'depth'),
        normal_root=os.path.join(dataset_root, 'normal'),
        anchor_root=os.path.join(dataset_root, 'anchors'),
        adj_path=os.path.join(dataset_root, 'smpl_adj_512.npy'),
        fps_indices_path=os.path.join(dataset_root, 'fps_indices_512.npy'),
        smplx_model_path='smplx_models/',
        resolution=config.data.resolution,
        sequences=getattr(config.data, 'sequences', None),
        num_epochs_per_epoch=1
    )
    
    # 3. Predict Single Item Process
    item = dataset[0] # Fetch one scene triplet
    view1, view2 = item['context']
    
    # Unsqueeze to simulate batch size of 1 and push to device
    view1['img'] = view1['img'].unsqueeze(0).cuda()
    view2['img'] = view2['img'].unsqueeze(0).cuda()
    
    # Tensors that aren't natively dicts need explicit batching + device pinning
    adj = item['smpl_adj'].cuda()
    anchors1 = view1['mesh_anchors'].unsqueeze(0).cuda()
    anchors2 = view2['mesh_anchors'].unsqueeze(0).cuda()

    print("Running forward feature pass utilizing Cross-Attention Priors...")
    pred1, pred2 = model.forward(view1, view2, adj, anchors1, anchors2)
    
    # Output Image Dimensions
    _, _, h, w = view1['img'].shape

    print("Rendering novel target view using Decoder Splatting...")
    # Wrap item back into a simulated dataloader batch
    batch = {
        'context': [view1, view2],
        'target': [item['target'][0]]
    }
    batch['target'][0]['camera_pose'] = batch['target'][0]['camera_pose'].unsqueeze(0).cuda()
    batch['target'][0]['camera_intrinsics'] = batch['target'][0]['camera_intrinsics'].unsqueeze(0).cuda()

    color, _ = model.decoder(batch, pred1, pred2, (h, w))

    # Visually Save Output
    pred_numpy = einops.rearrange(color[0, 0], 'c h w -> h w c').cpu().numpy()
    plt.imsave("inference_result.png", pred_numpy.clip(0, 1))
    print("Novel view fully synthesized and saved to `inference_result.png`!")

if __name__ == "__main__":
    # Ensure a basic config file specifies devices, batches, etc.
    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    infer_zero_shot(config, "dummy_path_for_now.ckpt")
