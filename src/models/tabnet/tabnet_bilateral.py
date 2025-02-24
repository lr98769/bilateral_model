import torch
from torch.nn import Module
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_group_matrix
from torchsummary import summary
from torch.nn import BCELoss
from torch.optim import Adam
from tqdm.auto import tqdm
import time
from torch.nn.utils import clip_grad_norm_

from src.pytorch_training.callbacks import HistoryRecorder, EarlyStoppingCallback, MetricCalculator
from src.configs.config import device
from src.pytorch_training.display_progress import print_split_epoch_metrics, plot_history

from src.preprocessing.pytorch_preprocessing import get_pytorch_split_dict
from src.pytorch_training.misc import set_seed_pytorch


class TabNetBilateralClassifier(Module):
    def __init__(
        self, 
        input_dim, output_dim, 
        n_d=8, n_a=8, cat_emb_dim=2,
        grouped_features=list(), lambda_sparse: float = 1e-3):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        group_matrix = create_group_matrix(grouped_features, input_dim)
        self.base_model = TabNet(
            n_d=n_d, n_a=n_a, cat_emb_dim=cat_emb_dim,
            input_dim=input_dim, output_dim=output_dim, group_attention_matrix=group_matrix)
        
    def forward(self, le, re):
        le_pred, le_M_loss = self.base_model(le)
        re_pred, re_M_loss = self.base_model(re)
        return torch.sigmoid(le_pred) * torch.sigmoid(re_pred), le_M_loss+re_M_loss
    
def ordinal_to_label(ordinal):
    # [[1, 0, 0, 0]] = 0
    # [[1, 1, 1, 1]] = 3
    class_num = (ordinal.round().sum(axis=-1)-1).type(torch.LongTensor)
    return class_num
    
def prediction(model: TabNetBilateralClassifier, dl, verbose=True):
    all_outputs =  None
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for (le_input, re_input), ordinal_output in pbar:
                output = model(le_input.to(device), re_input.to(device)) #(num_samples, num_outputs)
                if output is tuple:
                    ordinal_pred = output[0]
                else:
                    ordinal_pred = output
                ordinal_pred = ordinal_pred.detach().cpu()
                output = ordinal_to_label(ordinal_output)
                output_pred = ordinal_to_label(ordinal_pred)
                cur_batch_output = [le_input, re_input, ordinal_output, ordinal_pred, output, output_pred]
                num_elements = len(cur_batch_output)
                # If this is the first batch
                if all_outputs is None: 
                    all_outputs = cur_batch_output
                else:
                    for i in range(num_elements):
                        # all_outputs[i] = (num_samples, element_size)
                        # cur_batch_output[i] = (batch_size, element_size)
                        all_outputs[i] = torch.cat((all_outputs[i], cur_batch_output[i]), dim=0)
    return all_outputs #(, , , )

def train_tabnet_bilateral(
    model, train_dl, valid_dl, test_dl, 
    fp_model, fp_history=None, # Where to store trained model and history of training
    max_epochs=500, lr=0.001, weight_decay=0.1,  # Training parameters
    patience=5, metric_to_monitor = "acc", maximise=True, # For early stopping
    verbose=True
):
    metric_list = ["acc", "f1"]
    split_list = ["train", "valid"]
    
    model.train()
    model.to(device)
    
    loss_fn = BCELoss() # MSELoss()
    optimizer = Adam(model.base_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialise history recorder
    hr = HistoryRecorder(split_list, metric_list)
    
    # EarlyStopping callback
    es = EarlyStoppingCallback(patience, metric_to_monitor, fp_model, maximise=maximise)

    # Initialise metric calculator
    mc = MetricCalculator(metric_list)
    
    with tqdm(range(max_epochs), leave=False, disable=verbose) as epoch_pbar:
        for epoch in epoch_pbar:
            if epoch > 0:
                epoch_pbar.set_description(f"Valid {metric_to_monitor.capitalize()}: {val_score:.5f}")
            start = time.time()
            all_outputs = None
            position = 0 if verbose else 1 
            with tqdm(train_dl, leave=False, position=position, disable=not verbose) as pbar:
                for (le_x, re_x), y_ordinal in pbar:
                    le_x, re_x, y_ordinal = le_x.to(device), re_x.to(device), y_ordinal.to(device)
                    y_ordinal_pred, M_loss = model(le_x, re_x) # This is the ordinal output ()
                    y = ordinal_to_label(y_ordinal)
                    y_pred = ordinal_to_label(y_ordinal_pred)
                    
                    loss = loss_fn(y_ordinal_pred, y_ordinal)
                    loss = loss - model.lambda_sparse * M_loss
                    optimizer.zero_grad()
                    clip_grad_norm_(model.base_model.parameters(), 1)
                    loss.backward()
                    optimizer.step()
                    
                    cur_output_batch = [le_x, re_x, y_ordinal, y_ordinal_pred, y, y_pred]
                    cur_output_batch = [element.detach().cpu() for element in cur_output_batch]
                    num_outputs = len(cur_output_batch)
                    
                    if all_outputs is None: # First Batch
                        all_outputs = cur_output_batch
                    else:
                        for i in range(num_outputs):
                            all_outputs[i] = torch.concat((all_outputs[i], cur_output_batch[i]))
                    del le_x, re_x, y_ordinal, y_ordinal_pred
    
            # Calculate train metrics
            train_metric_dict = mc.calculate_metric_dict(all_outputs[4], all_outputs[5])
    
            # Calculate validation metrics
            valid_outputs = prediction(model, valid_dl, verbose)
            valid_metric_dict = mc.calculate_metric_dict(valid_outputs[4], valid_outputs[5])
    
            # Print epoch metrics
            epoch_time = time.time() - start
            if verbose:
                print(f"=== Epoch {epoch}, Time Taken: {epoch_time:.1f}s, Time Left: {epoch_time*(max_epochs-epoch-1):.1f}s ===")
                print_split_epoch_metrics(splitname="train", metric_dict=train_metric_dict)
                print_split_epoch_metrics(splitname="valid", metric_dict=valid_metric_dict)
    
            # Save metrics to visualise training history
            hr.record_epoch_metrics(splitname="train", metric_dict=train_metric_dict)
            hr.record_epoch_metrics(splitname="valid", metric_dict=valid_metric_dict)
    
            # Early Stopping
            val_score = valid_metric_dict[metric_to_monitor]
            if es.stop_training(val_score, epoch, model):
                break
            
    history = hr.get_history_dict()
    plot_history(history, split_list, metric_list, max_cols = 3, fp_history=fp_history)
    return history


def train_model_w_best_param(
    ModelClass, best_param, 
    data_dict, col_info, batch_size, eval_batch_size, 
    train_param_dict, train_model_func, seed, fp_model, fp_history,
    metric_to_monitor="auc", maximise=True, 
    
):
    set_seed_pytorch(seed)
    split_dict_pytorch = get_pytorch_split_dict(
        data_dict=data_dict, col_info=col_info, batch_size=batch_size, eval_batch_size=eval_batch_size
    )
    model = ModelClass(
        **best_param, input_dim=len(col_info["input_cols"]), output_dim=len(col_info['output_cols']))
    history = train_model_func(
        model=model, **split_dict_pytorch, 
        fp_model=fp_model, **train_param_dict, fp_history=fp_history, 
        metric_to_monitor=metric_to_monitor, maximise=maximise
    )
    model = torch.load(fp_model, weights_only=False)
    return model


import pandas as pd

def evaluate_baseline_bilteral_models(
    data_dfs, col_info, model, batch_size, eval_batch_size, seed
):
    # Set Seed
    set_seed_pytorch(seed)
    
    train_df = data_dfs["train_df"]
    valid_df = data_dfs["valid_df"]
    test_df = data_dfs["test_df"]

    # Get feature col information 
    input_cols = col_info["input_cols"]
    output_cols = col_info["output_cols"]
    le_label = col_info["le_label"]
    re_label = col_info["re_label"]

    # Preprocess data
    split_dict_pytorch = get_pytorch_split_dict(
        data_dict=data_dfs, col_info=col_info, batch_size=batch_size, eval_batch_size=eval_batch_size
    )
    pred_df = pd.concat([train_df, valid_df, test_df])
    pred_df["split"] = (
        ["train" for _ in range(len(train_df))] + ["valid" for _ in range(len(valid_df))] + ["test" for _ in range(len(test_df))])
    
    all_y_pred = []
    all_y = []
    for dl_name, dl in split_dict_pytorch.items():
        pred = prediction(model, dl, verbose=True)
        all_y_pred.append(pred[-1])
        all_y.append(pred[-2])
    all_y_pred = torch.cat(all_y_pred)
    all_y = torch.cat(all_y)
    pred_df["actual"] = all_y
    pred_df["pred"] = all_y_pred

    truth_array = all_y_pred==all_y
    pred_df["correct?"] = truth_array

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

    return pred_df, perf_df
        
        
