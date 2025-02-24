import pickle
import pandas as pd
from os import makedirs
from os.path import join, exists

def save_hyperparameters(best_hyperparameters, tuning_df, name, fp_checkpoint_folder, override=False):
    fp_cur_checkpoint = join(fp_checkpoint_folder, "hyperparameters", name)
    if exists(fp_cur_checkpoint):
        print("Checkpoint exist!")
        if not override:
            return
    else:
        makedirs(fp_cur_checkpoint)
    fp_tuning_df = join(fp_cur_checkpoint, "tuning.csv")
    fp_best_hyperparameters = join(fp_cur_checkpoint, "best_hyperparameter.pickle")
    tuning_df.to_csv(fp_tuning_df, index=False)
    with open(fp_best_hyperparameters, "wb") as f:
        pickle.dump(best_hyperparameters, f)
    print("Hyperparameters Saved!")
    
def load_hyperparameters(name, fp_checkpoint_folder):
    fp_cur_checkpoint = join(fp_checkpoint_folder, "hyperparameters", name)
    fp_tuning_df = join(fp_cur_checkpoint, "tuning.csv")
    fp_best_hyperparameters = join(fp_cur_checkpoint, "best_hyperparameter.pickle")
    tuning_df = pd.read_csv(fp_tuning_df)
    with open(fp_best_hyperparameters, "rb") as f:
        best_hyperparameters = pickle.load(f)
    return best_hyperparameters, tuning_df

def save_predictions(prediction_df, name, fp_checkpoint_folder):
    fp_cur_checkpoint = join(fp_checkpoint_folder, "predictions")
    fp_prediction_df = join(fp_cur_checkpoint, f"{name}.csv")
    if not exists(fp_cur_checkpoint):
        makedirs(fp_cur_checkpoint)
    prediction_df.to_csv(fp_prediction_df)
    print("Predictions Saved!")
    
def load_predictions(name, fp_checkpoint_folder):
    fp_cur_checkpoint = join(fp_checkpoint_folder, "predictions")
    fp_prediction_df = join(fp_cur_checkpoint, f"{name}.csv")
    return pd.read_csv(fp_prediction_df, index_col=0)