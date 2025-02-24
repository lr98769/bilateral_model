import numpy as np
import torch
import random

from src.preprocessing.pytorch_preprocessing import get_pytorch_split_dict
from src.configs.image_config import col_info



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers_until(model, layer_name):
    print("Freezing Weights:")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if layer_name in name:
            break
        print("-", name, "frozen!")


def set_seed_pytorch(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def reduce_dl_for_debugging(data_dfs, size, seed, batch_size, eval_batch_size, dl_func):
    small_data_dfs = {}
    for label, df in data_dfs.items():
        small_data_dfs[label] = df.sample(n=size, random_state=seed)
    set_seed_pytorch(seed)
    small_data_dls = get_pytorch_split_dict(
        data_dict=small_data_dfs, col_info=col_info, 
        batch_size=batch_size, eval_batch_size=eval_batch_size, shuffle_train=True,
        dl_func=dl_func
    )
    return small_data_dls

# from src.pytorch_training.misc import reduce_dl_for_debugging
# data_dls = reduce_dl_for_debugging(
#     data_dfs=data_dfs, size=100, seed=seed_no, 
#     batch_size=batch_size, eval_batch_size=eval_batch_size, 
#     dl_func=get_bc_image_dl)
