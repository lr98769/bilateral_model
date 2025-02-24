import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm
from src.misc import set_seed
from src.models.mlp.training import get_baseline_model
from src.models.mlp.training import train_baseline
from src.misc import get_mean_std_df

from src.preprocessing.tabular_preprocessing import preprocess_data_for_baseline


def evaluate_baseline(baseline_model, train_df, valid_df, test_df, col_info):
    # Get feature col information 
    input_cols = col_info["input_cols"]
    output_cols = col_info["output_cols"]
    le_label = col_info["le_label"]
    re_label = col_info["re_label"]
    
    # Preprocess data
    train_X, train_Y = preprocess_data_for_baseline(train_df, input_cols, output_cols, le_label, re_label)
    valid_X, valid_Y = preprocess_data_for_baseline(valid_df, input_cols, output_cols, le_label, re_label)
    test_X, test_Y = preprocess_data_for_baseline(test_df, input_cols, output_cols, le_label, re_label)

    # Evaluate ae model
    train_Y_pred = baseline_model.predict(train_X)
    valid_Y_pred = baseline_model.predict(valid_X)
    test_Y_pred = baseline_model.predict(test_X)
    def accuracy(actual, pred):
        actual_index = np.argmax(actual, axis=-1)
        pred_index = np.argmax(pred, axis=-1)
        return np.mean(actual_index==pred_index)
    train_acc = accuracy(train_Y, train_Y_pred)
    valid_acc = accuracy(valid_Y, valid_Y_pred)
    test_acc = accuracy(test_Y, test_Y_pred)
    def categorical_crossentropy(actual, pred):
        return np.mean(
            tf.keras.metrics.categorical_crossentropy(actual, pred))
    train_crossentropy = categorical_crossentropy(train_Y, train_Y_pred)
    valid_crossentropy = categorical_crossentropy(valid_Y, valid_Y_pred)
    test_crossentropy = categorical_crossentropy(test_Y, test_Y_pred)


    def convert_label_to_index(labels):
        return [np.argmax(cur_label, axis=-1) for cur_label in labels]
            
    def get_class_accuracies(actual, pred):
        value_list, label_list = [], []
        pred = pred > 0.5
        correctness = np.all(actual == pred, axis=-1)
        class_label = convert_label_to_index(actual)
        split_df = pd.DataFrame({"actual":class_label, "correct?":correctness})
        for i, class_label in enumerate(output_cols):
            class_df = split_df[split_df["actual"]==i]
            class_acc = class_df["correct?"].mean()
            value_list.append(class_acc)
            label_list.append(f"Accuracy {class_label}")
            output_dict = class_acc
            class_size = len(class_df)
            value_list.append(class_size/len(pred))
            label_list.append(f"{class_label} Proportion")
        return value_list, label_list

    train_class_acc = get_class_accuracies(train_Y, train_Y_pred)
    valid_class_acc = get_class_accuracies(valid_Y, valid_Y_pred)
    test_class_acc = get_class_accuracies(test_Y, test_Y_pred)
    
    perf_df = pd.DataFrame({
        "Train": [train_acc, train_crossentropy]+train_class_acc[0], 
        "Valid": [valid_acc, valid_crossentropy]+valid_class_acc[0], 
        "Test": [test_acc, test_crossentropy]+test_class_acc[0]})
    perf_df.index = ["Final Classifier Accuracy", "Final Classifier Crossentropy"]+ test_class_acc[1]
    perf_df = perf_df.sort_index().round(3)
    
    # Output predictions
    all_Y = np.concatenate((train_Y, valid_Y, test_Y), axis=0)
    all_Y_pred = np.concatenate((train_Y_pred, valid_Y_pred, test_Y_pred), axis=0)
    array = np.concatenate((all_Y, all_Y_pred), axis=1)
    colnames = output_cols + [final_col+"_pred" for final_col in output_cols]
    pred_df = pd.DataFrame(array, columns=colnames)
    split_labels = (
        ["train" for i in range(len(train_Y))] + ["valid" for i in range(len(valid_Y))] + 
        ["test" for i in range(len(test_Y))])
    pred_df["split"] = split_labels

    
    return perf_df, pred_df

def run_experiment_mlp_repetition(param, data_dfs, col_info, batch_size, repetitions, seed, dp):
    all_perf_dfs = []
    for cur_seed in tqdm(range(seed, seed+repetitions), total=repetitions):
        set_seed(cur_seed)
        baseline_model = get_baseline_model(
            num_input_cols=len(col_info["input_cols"])*2, 
            num_output_cols=len(col_info["output_cols"]),
            **param
        )
        train_baseline(
            baseline_model, **data_dfs, col_info=col_info,
            batch_size=batch_size, max_epochs=1000, patience=20, verbose=1, seed=cur_seed
        )
        perf_df, pred_df = evaluate_baseline(baseline_model, col_info=col_info, **data_dfs)
        all_perf_dfs.append(perf_df)
    return get_mean_std_df(all_perf_dfs, dp)