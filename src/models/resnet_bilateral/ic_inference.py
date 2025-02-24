from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from src.models.resnet_bilateral.inference import get_pred_perf_df
from src.configs.image_config import device
from src.models.resnet_bilateral.resnet_bilateral_model import ResNet18Bilateral
from src.label_converter import ImageLabelConverter

def predict_w_resnet_ic(model: ResNet18Bilateral, dl: DataLoader, verbose=True):
    all_outputs =  None
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for input_batch, output_batch in pbar:
                input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                output_pred = model.eye_prediction(input_batch) 
                cur_batch_output = [output_batch, output_pred]
                cur_batch_output = [element.detach().cpu() for element in cur_batch_output]
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

def evaluate_resnet_ic(model: ResNet18Bilateral, col_info: dict, dl: DataLoader, verbose=True):
    def add_label_to_list(cur_list, label):
        return [element+label for element in cur_list]
    
    num_classes = col_info["num_classes"]
    left_label, right_label = col_info["left_label"], col_info["right_label"]
    output_cols = col_info["ordinal_output_cols"]
    output_label_col = col_info["output_col"]
    logit_label = "_logit"
    left_true_cols = add_label_to_list(output_cols, left_label)
    right_true_cols = add_label_to_list(output_cols, right_label)
    left_logit_cols = add_label_to_list(left_true_cols, logit_label)
    right_logit_cols = add_label_to_list(right_true_cols, logit_label)
    left_true_col = output_label_col+left_label
    right_true_col = output_label_col+right_label
    pred_label = "_pred"
    left_pred_col = left_true_col+pred_label
    right_pred_col = right_true_col+pred_label
    both_label = "_both"
    both_true_col = output_label_col+both_label
    both_pred_col = both_true_col+pred_label
    label_converter = ImageLabelConverter(num_classes)
    
    # Get Prediction DF
    all_outputs = predict_w_resnet_ic(model, dl, verbose=verbose)
    true_label, logits = all_outputs[0], all_outputs[1]
    num_samples = round(len(true_label)/2)
    pred_df = pd.DataFrame(
        np.concatenate((true_label[:num_samples], true_label[num_samples:]), axis=1)
    , columns=left_true_cols+right_true_cols)
    pred_df[left_logit_cols] = logits[:num_samples]
    pred_df[right_logit_cols] = logits[num_samples:]
    
    pred_df[left_true_col] = label_converter.decode_labels(pred_df[left_true_cols].values)
    pred_df[right_true_col] = label_converter.decode_labels(pred_df[right_true_cols].values)
    pred_df[left_pred_col] = label_converter.decode_labels(pred_df[left_logit_cols].values)
    pred_df[right_pred_col] = label_converter.decode_labels(pred_df[right_logit_cols].values)
    
    # Make Both Eye Pred
    pred_df[both_true_col] = pred_df[[left_true_col, right_true_col]].values.max(axis=-1)
    pred_df[both_pred_col] = pred_df[[left_pred_col, right_pred_col]].values.max(axis=-1)
    
    # Evaluate
    perf_df = get_pred_perf_df(pred_df, col_info)
    
    return pred_df, perf_df

