import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sn
import re
from os.path import join

from src.misc import set_seed, create_folder
from src.preprocessing.tabular_preprocessing import preprocess_data_for_ae_ic_fc, generate_le_re_df


def train_valid_test_evaluator(metric_vec, split_vec):
    train_metric = np.mean(metric_vec[split_vec=="train"])
    valid_metric = np.mean(metric_vec[split_vec=="valid"])
    test_metric = np.mean(metric_vec[split_vec=="test"])
    return [train_metric, valid_metric, test_metric]

def evaluate_final_model(
    bilateral_model, train_df, valid_df, test_df, intermediate_col_dict, seed, fp_checkpoint_folder
):
    import time
    # Set Seed
    set_seed(seed)

    # Get feature col information from model
    input_cols = list(bilateral_model.input_cols.numpy().astype('str'))
    intermediate_cols = list(bilateral_model.intermediate_cols.numpy().astype('str'))
    output_cols = list(bilateral_model.output_cols.numpy().astype('str'))
    le_label = bilateral_model.le_label.numpy().decode("utf-8")
    re_label = bilateral_model.re_label.numpy().decode("utf-8")

    # Preprocess data
    all_df = pd.concat([train_df, valid_df, test_df])
    split_vec = np.concatenate((
        np.full(len(train_df), "train"), np.full(len(valid_df), "valid"), np.full(len(test_df), "test")
    ))
    all_df["Split"] = split_vec
    all_fc_X, all_ic, all_fc_Y = preprocess_data_for_ae_ic_fc(
        all_df, input_cols=input_cols, intermediate_cols=intermediate_cols, output_cols=output_cols,
        le_label=le_label, re_label=re_label
    )

    # Get Predictions
    prediction_dict = bilateral_model.predict_everything(all_fc_X[0], all_fc_X[1])
    
    # Create prediction df
    decoder_df = generate_le_re_df(
        le_mat=prediction_dict["decoder1"], re_mat=prediction_dict["decoder2"], 
        col_list=input_cols, le_label=le_label, re_label=re_label, suffix="reconstructed", df=all_df
    )
    ic_df = generate_le_re_df(
        le_mat=prediction_dict["ic1"], re_mat=prediction_dict["ic2"], 
        col_list=intermediate_cols, le_label=le_label, re_label=re_label, suffix="ic", df=all_df
    )
    fc_df = generate_le_re_df(
        le_mat=prediction_dict["fc1"], re_mat=prediction_dict["fc2"], 
        col_list=output_cols, le_label=le_label, re_label=re_label, suffix="fc", df=all_df
    )
    pred_output_cols = [f"{col}_pred" for col in output_cols]
    fc_df[pred_output_cols] = prediction_dict["output"]
    pred_df = pd.concat([all_df, decoder_df, ic_df, fc_df], axis=1) 

    # Prediction Performance
    prediction_performance_dict = {}
    
    # - Reconstruction
    le_re_input_cols = (
        [feat_col + f" {le_label}" for feat_col in input_cols] + 
        [feat_col + f" {re_label}" for feat_col in input_cols])
    decoder_metric_vec = np.mean(np.square(decoder_df.values-all_df[le_re_input_cols].values), -1)
    prediction_performance_dict["Reconstruction Error"] = train_valid_test_evaluator(
        metric_vec=decoder_metric_vec, split_vec=split_vec)
    
    # - Intermediate Classifier
    le_re_intermediate_cols = (
        [feat_col + f" {le_label}" for feat_col in intermediate_cols] + 
        [feat_col + f" {re_label}" for feat_col in intermediate_cols])
    ic_metric_vec = tf.keras.metrics.binary_crossentropy(
        all_df[le_re_intermediate_cols].values, ic_df.values
    )

    ic_df["Split"] = split_vec
    evaluate_per_class_statistics(
        pred_df=ic_df, actual_df=all_df[le_re_intermediate_cols+["Split"]], 
        intermediate_col_dict=intermediate_col_dict, fp_checkpoint_folder=fp_checkpoint_folder
    )
    evaluate_per_class_statistics_single(
        pred_df=ic_df, actual_df=all_df[le_re_intermediate_cols+["Split"]], 
        intermediate_col_dict=intermediate_col_dict, fp_checkpoint_folder=fp_checkpoint_folder
    )
    evaluate_per_class_statistics(
        pred_df=ic_df, actual_df=all_df[le_re_intermediate_cols+["Split"]], 
        intermediate_col_dict=intermediate_col_dict, split="valid", fp_checkpoint_folder=fp_checkpoint_folder
    )
    evaluate_per_class_statistics_single(
        pred_df=ic_df, actual_df=all_df[le_re_intermediate_cols+["Split"]], 
        intermediate_col_dict=intermediate_col_dict, split="valid", fp_checkpoint_folder=fp_checkpoint_folder
    )
    evaluate_per_class_statistics(
        pred_df=ic_df, actual_df=all_df[le_re_intermediate_cols+["Split"]], 
        intermediate_col_dict=intermediate_col_dict,split="train", fp_checkpoint_folder=fp_checkpoint_folder
    )
    evaluate_per_class_statistics_single(
        pred_df=ic_df, actual_df=all_df[le_re_intermediate_cols+["Split"]], 
        intermediate_col_dict=intermediate_col_dict,split="train", fp_checkpoint_folder=fp_checkpoint_folder
    )

    # May need to change this since it is no longer binary
    prediction_performance_dict["Intermediate Binary Crossentropy"] = train_valid_test_evaluator(
        metric_vec=ic_metric_vec, split_vec=split_vec)
    # - Final Classifier
    fc_metric_vec = tf.keras.metrics.binary_crossentropy(
        all_df[output_cols].values, fc_df[pred_output_cols].values
    )
    prediction_performance_dict["Final Binary Crossentropy"] = train_valid_test_evaluator(
        metric_vec=fc_metric_vec, split_vec=split_vec)
    prediction_performance_df = pd.DataFrame(prediction_performance_dict)
    prediction_performance_df.index = ["Train", "Validation", "Test"]
    
    return pred_df, prediction_performance_df

def evaluate_per_class_statistics(pred_df, actual_df, intermediate_col_dict, fp_checkpoint_folder, split="test"):
    fp_output_folder = join(fp_checkpoint_folder, split)
    create_folder(fp_output_folder)
    pred_df = pred_df[pred_df["Split"] == split]
    actual_df = actual_df[actual_df["Split"] == split]
    fig, axes = plt.subplots(1, len(intermediate_col_dict), figsize=(2*len(intermediate_col_dict), 2.5), dpi=300)
    # cbar_ax = fig.add_axes([0, 0, 1587, 1587]) # [xmin, ymin, width, and height]
    from sklearn.metrics import confusion_matrix
    for i, (condition, class_list) in enumerate(intermediate_col_dict.items()):
        all_pred, all_actual = [], []
        for eye in ["LE", "RE"]:
            class_col_list = [f"{condition}_{class_name} {eye}" for class_name in class_list]
            class_col_list_pred = [f"{condition}_{class_name} {eye}_ic" for class_name in class_list]
            pred_array = pred_df[class_col_list_pred].values.argmax(axis=-1)
            actual_array = actual_df[class_col_list].values.argmax(axis=-1)
            all_pred.append(pred_array)
            all_actual.append(actual_array)
        all_pred = np.concatenate(all_pred, axis=0)
        all_actual = np.concatenate(all_actual, axis=0)
        cm = confusion_matrix(all_actual, all_pred, labels=list(range(len(class_list))))
        def process_label(label):
            import re
            label = label.replace("_", " ")
            return re.sub(r'([a-z])([A-Z])', r'\1 \2', label)
            
        new_class_list = [process_label(classname) for classname in class_list]
        df_cm = pd.DataFrame(cm)
        df_cm.columns = new_class_list
        df_cm.index = new_class_list
        display(df_cm)
        
        sn.set(font_scale=0.5) # for label size
        sn.heatmap(df_cm, ax=axes[i], cmap="Blues", annot=True, fmt='g', vmin=0, vmax=1587, cbar=False)
        axes[i].set_title(condition[:-1], fontsize=14)
        axes[i].set_xlabel('Predicted', fontsize=8)
        axes[i].set_ylabel('Actual', fontsize=8)
    plt.tight_layout()
    plt.savefig(join(fp_output_folder, f"all_confusion_matrices_{split}.jpg"))
    plt.show()

def evaluate_per_class_statistics_single(
    pred_df, actual_df, intermediate_col_dict, fp_checkpoint_folder, split="test"):
    fp_output_folder = join(fp_checkpoint_folder, split)
    create_folder(fp_output_folder)
    pred_df = pred_df[pred_df["Split"] == split]
    actual_df = actual_df[actual_df["Split"] == split]
    
    # cbar_ax = fig.add_axes([0, 0, 1587, 1587]) # [xmin, ymin, width, and height]
    from sklearn.metrics import confusion_matrix
    for i, (condition, class_list) in enumerate(intermediate_col_dict.items()):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=300)
        all_pred, all_actual = [], []
        for eye in ["LE", "RE"]:
            class_col_list = [f"{condition}_{class_name} {eye}" for class_name in class_list]
            class_col_list_pred = [f"{condition}_{class_name} {eye}_ic" for class_name in class_list]
            pred_array = pred_df[class_col_list_pred].values.argmax(axis=-1)
            actual_array = actual_df[class_col_list].values.argmax(axis=-1)
            all_pred.append(pred_array)
            all_actual.append(actual_array)
        all_pred = np.concatenate(all_pred, axis=0)
        all_actual = np.concatenate(all_actual, axis=0)
        cm = confusion_matrix(all_actual, all_pred, labels=list(range(len(class_list))))
        def process_label(label):
            label = label.replace("_", " ")
            return re.sub(r'([a-z])([A-Z])', r'\1 \2', label)
            
        new_class_list = [process_label(classname) for classname in class_list]
        df_cm = pd.DataFrame(cm)
        df_cm.columns = new_class_list
        df_cm.index = new_class_list
        display(df_cm)
        
        sn.set(font_scale=0.5) # for label size
        sn.heatmap(df_cm, ax=ax, cmap="Blues", annot=True, fmt='g', vmin=0, vmax=1587, cbar=False)
        ax.set_title(condition[:-1], fontsize=14)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual', fontsize=8)
        plt.savefig(join(fp_output_folder, f"{condition}_confusion_matrices_{split}.jpg"), bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    
from tqdm.auto import tqdm
from src.models.bilateral.model import BilateralModel
from src.models.bilateral.training.training_ae import train_ae, evaluate_ae
from src.models.bilateral.training.training_ic import train_ic, evaluate_ic
from src.models.bilateral.training.training_fc import train_fc, evaluate_fc
from src.models.bilateral.training.training import transfer_ae, transfer_ae_n_ic
from src.misc import get_mean_std_df
    
def run_experiment_bilateral_repetition(
    ae_param, ic_param, fc_param, data_dfs, col_info, batch_size, repetitions, seed, dp):
    ae_perf_dfs, ic_perf_dfs, fc_perf_dfs = [], [], []
    with tqdm(range(seed, seed+repetitions), total=repetitions) as pbar:
        for cur_seed in pbar:
            # Train ae
            pbar.set_description("Training AE")
            set_seed(cur_seed)
            ae_bilateral_model = BilateralModel(**col_info, **ae_param)
            train_ae(
                ae_bilateral_model, **data_dfs,
                batch_size=batch_size, max_epochs=1000, patience=20, verbose=0, seed=cur_seed
            )
            ae_perf_df, ae_pred_df = evaluate_ae(ae_bilateral_model, **data_dfs)
            ae_perf_dfs.append(ae_perf_df)
            # Train IC
            pbar.set_description("Training IC")
            set_seed(cur_seed)
            ic_bilateral_model = BilateralModel(**col_info, **ic_param)
            ic_bilateral_model = transfer_ae(base_bilateral_model=ae_bilateral_model, new_bilateral_model=ic_bilateral_model)
            train_ic(
                ic_bilateral_model, **data_dfs,
                batch_size=batch_size, max_epochs=10000, patience=20, verbose=0, seed=cur_seed
            )
            ic_perf_df, ic_pred_df = evaluate_ic(ic_bilateral_model, **data_dfs)
            ic_perf_dfs.append(ic_perf_df)
            # Train FC
            pbar.set_description("Training FC")
            set_seed(cur_seed)
            fc_bilateral_model = BilateralModel(**col_info, **fc_param,)
            fc_bilateral_model = transfer_ae_n_ic(base_bilateral_model=ic_bilateral_model, new_bilateral_model=fc_bilateral_model)
            train_fc(
                fc_bilateral_model, **data_dfs,
                batch_size=batch_size, max_epochs=1000, patience=20, verbose=0, seed=cur_seed
            )
            fc_perf_df, fc_pred_df = evaluate_fc(
                fc_bilateral_model, **data_dfs
            )
            fc_perf_dfs.append(fc_perf_df)
    return (
        get_mean_std_df(ae_perf_dfs, dp), 
        get_mean_std_df(ic_perf_dfs, dp), 
        get_mean_std_df(fc_perf_dfs, dp)
    )
