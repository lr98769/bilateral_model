from sklearn.model_selection import ParameterGrid
from time import time
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from src.misc import set_seed
from src.models.bilateral.model import BilateralModel

def tune_model(
    train_df, valid_df, test_df, 
    input_cols, intermediate_col_dict, output_cols, le_label, re_label,
    hyperparameter_list_dict, train_function,  seed, batch_size=32, max_epochs=1000, patience=20, verbose=1,
    base_bilateral_model=None, transfer_function=None
):
    
    # Parameters in hyperparameter_list_dict:
    # - ae_width, intermediate_classifier_width, final_classifier_width, 
    # - rho=0.1, beta=1
    tuning_list = []
    param_grid = list(ParameterGrid(hyperparameter_list_dict))
    val_loss_col = "valid_loss"
    pb = tqdm(param_grid, total=len(param_grid))
    for param_dict in pb:
        pb.set_description(f"{param_dict}")
        start = time()
        set_seed(seed)
        bilateral_model = BilateralModel(
            input_cols=input_cols, intermediate_col_dict=intermediate_col_dict, output_cols=output_cols, 
            le_label=le_label, re_label=re_label,
            **param_dict
        )
        # Transfer trained components from one model to another
        if base_bilateral_model and transfer_function:
            bilateral_model = transfer_function(
                base_bilateral_model=base_bilateral_model, new_bilateral_model=bilateral_model)
        valid_loss, valid_loss_wo_sparsity, best_epoch = train_function(
            bilateral_model, train_df, valid_df, test_df, seed=seed,
            batch_size=batch_size, max_epochs=max_epochs, patience=patience, verbose=verbose
        )
        hp_dict = param_dict.copy()
        hp_dict["valid_loss"] = valid_loss
        hp_dict["valid_loss_wo_sparsity"] = valid_loss_wo_sparsity
        hp_dict["best_epoch"] = best_epoch
        hp_dict["training_time/s"] = time()-start
        tuning_list.append(hp_dict)
    tuning_df = pd.DataFrame(tuning_list)
    best_hyperparameter_index = np.argmin(tuning_df[val_loss_col])
    tuning_df["best_hyperparameter"] = tuning_df.index == best_hyperparameter_index
    best_hyperparameter = param_grid[best_hyperparameter_index]
    return best_hyperparameter, tuning_df