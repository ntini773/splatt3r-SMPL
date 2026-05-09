import os
import torch
import argparse
import lightning
from datetime import datetime
from models.pipeline.pipeline import OurPipeline

# from dataset.webdata_loader import build_web_tracked_data
from dataset.webdata_loader import build_web_tracked_data
from utils.general_utils import (
    ConfigDict, rtqdm, device_parser, 
    calc_parameters, biuld_logger,add_extra_cfgs
)
import shutil

import sys

# timestamp = datetime.now().strftime("%Y%m%d_%H")
# log_path=os.path.join('outputs',  timestamp)
# os.makedirs(log_path, exist_ok=True)
# log_file = os.path.join(log_path, "train.log")
# sys.stdout = open(log_file, "w")
# sys.stderr = sys.stdout




def train( config_name, ehm_basemodel, devices, debug=False):
    # build config
    # if torch.distributed.is_available():
    #     torch.distributed.init_process_group(backend="nccl")
    meta_cfg = ConfigDict(
        model_config_path=os.path.join('configs', f'{config_name}.yaml')
    )
    meta_cfg = add_extra_cfgs(meta_cfg)
    lightning.fabric.seed_everything(10)
    target_devices = device_parser(devices)
    init_iter = 1
    print(str(meta_cfg))
    # setup model and optimizer



    # load dataset
    train_dataset = build_web_tracked_data(cfg_dataset=meta_cfg.DATASET, split='train')
    val_dataset = build_web_tracked_data(cfg_dataset=meta_cfg.DATASET, split='valid')


    timestamp = datetime.now().strftime("%Y%m%d_%H")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=meta_cfg.TRAIN.batch_size, num_workers=1, # shuffle=True,#
    ) # meta_cfg.TRAIN.batch_size
    val_dataloader = torch.utils.data.DataLoader( 
        val_dataset, batch_size=1, num_workers=1,  # shuffle=False,
    ) # meta_cfg.TRAIN.batch_size
    
    _dump_path=os.path.join('outputs',  timestamp)
    os.makedirs(_dump_path, exist_ok=True)
    shutil.copy(os.path.join('configs', f'{config_name}.yaml'), os.path.join(_dump_path, 'config.yaml'))

    ehm_model = OurPipeline(meta_cfg, train_dataloader, val_dataloader, target_devices)


    if ehm_basemodel is not None:
        assert os.path.exists(ehm_basemodel), f'Base model not found: {ehm_basemodel}.'
        _state=torch.load(ehm_basemodel, map_location='cpu', weights_only=True)
        ehm_model.backbone.load_state_dict(_state['backbone'], strict=False)
        ehm_model.head.load_state_dict(_state['head'], strict=False)
        print('Load base model from: {}.'.format(ehm_basemodel))

    ehm_model.run_fit()







if __name__ == "__main__":
    # import warnings
    # from tqdm.std import TqdmExperimentalWarning
    # warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    # build args  
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c', required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    parser.add_argument('--ehm_model', default=None, type=str) # 
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))
    # launch
    torch.set_float32_matmul_precision('high')
    train(args.config_name, args.ehm_model,  args.devices, args.debug)