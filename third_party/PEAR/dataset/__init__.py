from .data_loader import TrackedData,TrackedData_infer
from .data_loader2 import TrackedData2
from .data_loader3 import TrackedData3
# from dataset.data_loder2 import TrackedData2

def build_dataset(data_cfg, split,):

    return TrackedData(data_cfg, split)



def build_dataset_2(data_cfg, split,):

    return TrackedData2(data_cfg, split)

def build_dataset_3(data_cfg, split,):

    return TrackedData3(data_cfg, split)


