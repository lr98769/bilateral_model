import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from time import time
from tqdm.auto import tqdm

from src.misc import set_seed
from src.models.mlp.training import get_baseline_model

def tune_model(
    train_df, valid_df, test_df, col_info, 
    hyperparameter_list_dict, train_function, seed, batch_size=32, max_epochs=1000, patience=20, verbose=1, 
    base_bilateral_model=None, transfer_function=None
):  
    # Parameters in hyperparameter_list_dict:
    # - ae_width, intermediate_classifier_width, final_classifier_width, 
    # - rho=0.1, beta=1
    tuning_list = []
    param_grid = list(ParameterGrid(hyperparameter_list_dict))
    val_loss_col = "valid_loss"
    pb = tqdm(param_grid)
    for param_dict in pb:
        pb.set_description(f"{param_dict}")
        start = time()
        set_seed(seed)
        baseline_model = get_baseline_model(
            num_input_cols=len(col_info["input_cols"])*2, num_output_cols=len(col_info["output_cols"]),
            **param_dict
        )
        valid_loss, best_epoch = train_function(
            baseline_model, train_df, valid_df, test_df, col_info=col_info, 
            batch_size=batch_size, max_epochs=max_epochs, patience=patience, verbose=verbose, seed=seed
        )
        hp_dict = param_dict.copy()
        hp_dict["valid_loss"] = valid_loss
        hp_dict["best_epoch"] = best_epoch
        hp_dict["training_time/s"] = time()-start
        tuning_list.append(hp_dict)
    tuning_df = pd.DataFrame(tuning_list)
    best_hyperparameter_index = np.argmin(tuning_df[val_loss_col])
    tuning_df["best_hyperparameter"] = tuning_df.index == best_hyperparameter_index
    best_hyperparameter = param_grid[best_hyperparameter_index]
    return best_hyperparameter, tuning_df