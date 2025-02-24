import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import time
from sklearn.model_selection import ParameterGrid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from src.misc import get_mean_std_df, set_seed
from src.preprocessing.tabular_preprocessing import preprocess_data_for_scikitlearn


def tune_scikitlearn_classifiers(train_df, valid_df, test_df, col_info, Classifier, param_grid, seed):
    acc_col, best_col, time_col = "val_acc", "best", "time/s"
    tuned_params = list(param_grid.keys())
    tuning_df = []
    param_list = list(ParameterGrid(param_grid))
    for param in tqdm(param_list):
        clf, pred_df, perf_df, time_taken = train_n_evaluate_scikitlearn_classifiers(
            train_df, valid_df, test_df, col_info, Classifier, seed, param, return_time_taken=True
        )
        cur_acc = perf_df.loc["Accuracy", "valid"]
        result_dict = param.copy()
        result_dict[acc_col] = cur_acc 
        result_dict[time_col] = time_taken
        tuning_df.append(result_dict)
    tuning_df = pd.DataFrame(tuning_df)
    tuning_df[best_col] = False
    best_idx = tuning_df[acc_col].argmax()
    tuning_df[best_col].iloc[best_idx] = True
    # Get best param
    best_params = tuning_df.iloc[best_idx][tuned_params].to_dict()
    return tuning_df, best_params

def train_n_evaluate_scikitlearn_classifiers(
    train_df, valid_df, test_df, col_info, Classifier, seed, param, return_time_taken=False
):
    # Set Seed
    set_seed(seed)

    # Get feature col information 
    input_cols = col_info["input_cols"]
    output_cols = col_info["output_cols"]
    le_label = col_info["le_label"]
    re_label = col_info["re_label"]

    # Preprocess data
    train_X, train_Y, train_y = preprocess_data_for_scikitlearn(train_df, input_cols, output_cols, le_label, re_label)
    valid_X, valid_Y, valid_y = preprocess_data_for_scikitlearn(valid_df, input_cols, output_cols, le_label, re_label)
    test_X, test_Y, test_y = preprocess_data_for_scikitlearn(test_df, input_cols, output_cols, le_label, re_label)

    # Train_Model
    if (Classifier is QuadraticDiscriminantAnalysis) or (Classifier is TabNetClassifier):
        clf = Classifier(**param)
    else:
        clf = Classifier(**param, random_state=seed)
    
    if (Classifier is TabNetClassifier):
        start = time.time()
        clf.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], max_epochs=1000, batch_size=16)
    else:
        start = time.time()
        clf.fit(train_X, train_y)
    time_taken = time.time()-start

    pred_df = pd.concat([train_df, valid_df, test_df])
    pred_df["split"] = (
        ["train" for _ in range(len(train_df))] + ["valid" for _ in range(len(valid_df))] + ["test" for _ in range(len(test_df))])

    all_X = np.concatenate((train_X, valid_X, test_X), axis=0)
    all_y = np.concatenate((train_y, valid_y, test_y), axis=0)
    
    all_y_pred = clf.predict(all_X)
    pred_df["actual"] = all_y
    pred_df["pred"] = all_y_pred

    truth_array = all_y_pred==all_y
    pred_df["correct?"] = truth_array
    train_acc = pred_df[pred_df["split"]=="train"]["correct?"].mean()
    valid_acc = pred_df[pred_df["split"]=="valid"]["correct?"].mean()
    test_acc = pred_df[pred_df["split"]=="test"]["correct?"].mean()

    def get_stats(pred_df, split):
        split_df = pred_df[pred_df["split"]==split]
        overall_accuracy = split_df["correct?"].mean()
        output_dict = {"Accuracy": overall_accuracy}
        for i, class_label in enumerate(output_cols):
            class_df = split_df[split_df["actual"]==i]
            class_acc = class_df["correct?"].mean()
            output_dict[f"Accuracy {class_label} "] = class_acc
            class_size = len(class_df)
            output_dict[f"{class_label} Proportion"] = class_size/len(pred_df) # Some error here
        return output_dict

    splits = ["train", "valid", "test"]
    perf_df = pd.DataFrame([get_stats(pred_df, split)for split in splits], index=splits).T
    perf_df = perf_df.sort_index().round(3)

    # pref_df = pd.DataFrame({"Train": [train_acc],"Valid": [valid_acc], "Test": [test_acc]}, index=["Accuracy"])

    if return_time_taken:
        return clf, pred_df, perf_df, time_taken
    else:
        return clf, pred_df, perf_df
    
def run_experiment_scikitlearn_repetition(
    param, data_dfs, col_info, classifier, repetitions, seed, dp):
    all_perf_dfs = []
    for cur_seed in tqdm(range(seed, seed+repetitions), total=repetitions):
        model, pred_df, perf_df = train_n_evaluate_scikitlearn_classifiers(
            param=param, **data_dfs, col_info=col_info, Classifier=classifier, seed=cur_seed
        )
        all_perf_dfs.append(perf_df)
    return get_mean_std_df(all_perf_dfs, dp)
    
    
        