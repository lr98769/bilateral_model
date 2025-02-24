from torch.utils.data import DataLoader
import pandas as pd

from src.models.resnet.resnet_model import ResNet18
from src.models.resnet.resnet_inference import predict_w_resnet
from sklearn.metrics import accuracy_score
from src.models.resnet_bilateral.inference import get_pred_perf_df

def evaluate_resnet(model: ResNet18, col_info: dict, dl: DataLoader, verbose=True):
    num_classes = col_info["num_classes"]
    left_label, right_label = col_info["left_label"], col_info["right_label"]
    output_col = col_info["output_col"]
    logit_label = "_logit"
    left_true_col, right_true_col = output_col+left_label, output_col+right_label
    left_logit_cols = [left_true_col+f"_{i}"+logit_label for i in range(num_classes)]
    right_logit_cols = [right_true_col+f"_{i}"+logit_label for i in range(num_classes)]
    pred_label = "_pred"
    left_pred_col = left_true_col+pred_label
    right_pred_col = right_true_col+pred_label
    both_label = "_both"
    both_true_col = output_col+both_label
    both_pred_col = both_true_col+pred_label
    
    # Get Prediction DF
    all_outputs = predict_w_resnet(model, dl, verbose=verbose)
    true_label, logits = all_outputs[0], all_outputs[1]
    num_samples = round(len(true_label)/2)
    pred_df = pd.DataFrame({
        left_true_col: true_label[:num_samples],
        right_true_col: true_label[num_samples:],
    })
    pred_df[left_logit_cols] = logits[:num_samples]
    pred_df[right_logit_cols] = logits[num_samples:]
    pred_df[left_pred_col] = pred_df[left_logit_cols].values.argmax(axis=-1)
    pred_df[right_pred_col] = pred_df[right_logit_cols].values.argmax(axis=-1)
    
    # Make Both Eye Pred
    pred_df[both_true_col] = pred_df[[left_true_col, right_true_col]].values.max(axis=-1)
    pred_df[both_pred_col] = pred_df[[left_pred_col, right_pred_col]].values.max(axis=-1)
    
    # Evaluate
    perf_df = get_pred_perf_df(pred_df, col_info)
    # # - Get accuracy
    # perf_df = {
    #     "Acc": accuracy_score(y_true=pred_df[both_true_col], y_pred=pred_df[both_pred_col])
    # }
    # # - Get Class Accuracy
    # for i in range(num_classes):
    #     class_df = pred_df[pred_df[both_true_col]==i]
    #     perf_df[f"Class {i} Acc"] = accuracy_score(
    #         y_true=class_df[both_true_col], y_pred=class_df[both_pred_col],
    #     )
    # perf_df = pd.DataFrame([perf_df])
    
    return pred_df, perf_df

