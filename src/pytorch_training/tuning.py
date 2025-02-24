from src.preprocessing.pytorch_preprocessing import get_pytorch_split_dict
from src.pytorch_training.misc import set_seed_pytorch
from scipy.stats import pearsonr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
import time


def tune_model(
    ModelClass, param_grid, 
    data_dict, col_info, 
    batch_size, eval_batch_size, 
    train_param_dict, train_model_func, seed, 
    fp_model, fp_history,
    metric_to_monitor = "auc", maximise=True, 
    pytorch_split_dict_func=get_pytorch_split_dict
):
    
    split_to_monitor = "valid"
    display_params = [key for key, val in param_grid.items() if len(val)>1]
    
    metric_label = f"{split_to_monitor.capitalize()} {metric_to_monitor.capitalize()}"
    tuning_df_list = []
    parameter_list = list(ParameterGrid(param_grid))
    pbar = tqdm(parameter_list, total=len(parameter_list)) # , position=0
    best_score = -np.inf if maximise else np.inf
    val_score = None
    for param_dict in pbar:
        display_param_dict = {key: val for key, val in param_dict.items() if key in display_params}
        print(display_param_dict)
        if val_score is not None:
            pbar.set_description(f"Current Param: {display_param_dict}, Best {metric_label}: {best_score:.5f}, Lastest {metric_label}: {val_score:.5f}")
        else:
            pbar.set_description(f"Current Param: {display_param_dict}")
        set_seed_pytorch(seed)
        split_dict_pytorch = pytorch_split_dict_func(
            data_dict=data_dict, col_info=col_info, batch_size=batch_size, eval_batch_size=eval_batch_size
        )
        model = ModelClass(
            **param_dict, 
            input_dim=len(col_info["input_cols"]), output_dim=len(col_info['output_cols']))
        start = time.time()
        history = train_model_func(
            model=model, **split_dict_pytorch, 
            fp_model=fp_model, **train_param_dict, fp_history=fp_history, 
            metric_to_monitor=metric_to_monitor, maximise=maximise
        )
        best_metrics = get_best_val(history, split_to_monitor, metric_to_monitor, maximise=maximise)
        cur_param_dict = param_dict.copy()
        cur_param_dict.update(best_metrics)
        cur_param_dict["Time/s"] = time.time()-start
        tuning_df_list.append(cur_param_dict)

        val_score = cur_param_dict[metric_label]
        if maximise:
            if val_score > best_score:
                best_score = val_score
        else:
            if val_score < best_score:
                best_score = val_score
        
    tuning_df = pd.DataFrame(tuning_df_list)
    if maximise:
        best_param_idx = tuning_df[metric_label].idxmax()
    else:
        best_param_idx = tuning_df[metric_label].idxmin()
    tuning_df["best_hyperparameter"] = [
        True if i==best_param_idx else False for i in range(len(tuning_df))]
    best_param = parameter_list[best_param_idx]
    return tuning_df, best_param

def get_best_val(history, split_to_monitor, metric_to_monitor, maximise=True):
    if maximise:
        best_epoch = np.argmax(history[split_to_monitor][metric_to_monitor])
    else:
        best_epoch = np.argmin(history[split_to_monitor][metric_to_monitor])
    best_metrics = {"Epochs": best_epoch}
    for split, split_history_dict in history.items():
        for metric, metric_list in split_history_dict.items():
            best_metrics[f"{split.capitalize()} {metric.capitalize()}"] = metric_list[best_epoch]
    return best_metrics

def show_tuning_graphs(tuning_df, hp_list):
    fig, axes = plt.subplots(1, len(hp_list), dpi=300, figsize=(len(hp_list)*2, 2*1))
    if len(hp_list) == 1:
        axes = [axes]
    for i, hp in enumerate(hp_list):
        axes[i].scatter(tuning_df[hp], tuning_df["Valid Auc"])
        axes[i].set_xlabel(hp)
        axes[i].set_ylabel("Valid Auc")
    plt.tight_layout()
