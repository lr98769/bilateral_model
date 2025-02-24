from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

from src.configs.image_config import device
from src.models.resnet_bilateral.resnet_bilateral_model import ResNet18Bilateral
from src.label_converter import ImageLabelConverter
from src.models.resnet_bilateral.inference import get_pred_perf_df

def predict_w_resnet_bc(model: ResNet18Bilateral, dl: DataLoader, verbose=True):
    all_outputs =  None
    model.eval()
    model.to(device)
    num_batches = len(dl)
    with torch.no_grad():
        with tqdm(dl, leave=False, position=0, total=num_batches, disable=not verbose) as pbar:
            for (left_batch, right_batch), output_batch in pbar:
                left_batch, right_batch = left_batch.to(device), right_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred = model.patient_prediction(left_batch, right_batch) 
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

def evaluate_resnet_bc(model: ResNet18Bilateral, col_info: dict, dl: DataLoader, verbose=True):
    def add_label_to_list(cur_list, label):
        return [element+label for element in cur_list]
    
    num_classes = col_info["num_classes"]
    output_cols = col_info["ordinal_output_cols"]
    output_label_col = col_info["output_col"]
    both_label = "_both"
    
    logit_label = "_logit"
    true_cols = add_label_to_list(output_cols, both_label)
    logit_cols = add_label_to_list(true_cols, logit_label)
    
    pred_label = "_pred"
    true_label_col = output_label_col+both_label
    pred_label_col = true_label_col+pred_label
    label_converter = ImageLabelConverter(num_classes)
    
    # Get Prediction DF
    all_outputs = predict_w_resnet_bc(model, dl, verbose=verbose)
    true_label, logits = all_outputs[0], all_outputs[1]
    pred_df = pd.DataFrame(true_label, columns=true_cols)
    pred_df[logit_cols] = logits
    
    # Make Both Eye Pred
    pred_df[true_label_col] = label_converter.decode_labels(pred_df[true_cols].values)
    pred_df[pred_label_col] = label_converter.decode_labels(pred_df[logit_cols].values)
    
    # Evaluate
    perf_df = get_pred_perf_df(pred_df, col_info)
    
    return pred_df, perf_df
