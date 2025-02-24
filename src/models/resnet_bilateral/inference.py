import pandas as pd
from sklearn.metrics import accuracy_score


def get_pred_perf_df(pred_df, col_info):
    num_classes = col_info["num_classes"]
    output_label_col = col_info["output_col"]
    both_label = "_both"
    pred_label = "_pred"

    both_true_col = output_label_col+both_label
    both_pred_col = both_true_col+pred_label

    # - Get accuracy
    perf_df = {
        "Acc": accuracy_score(y_true=pred_df[both_true_col], y_pred=pred_df[both_pred_col])
    }
    # - Get Class Accuracy
    avg_class_accuracy = 0
    for i in range(num_classes):
        class_df = pred_df[pred_df[both_true_col]==i]
        perf_df[f"Class {i} Acc"] = accuracy_score(
            y_true=class_df[both_true_col], y_pred=class_df[both_pred_col],
        )
        avg_class_accuracy += perf_df[f"Class {i} Acc"]
    perf_df["Average Class Acc"] = avg_class_accuracy/num_classes
    # - Accuracy without Class 0
    cur_df = pred_df[pred_df[both_true_col]!=0]
    perf_df["Acc W/O Class 0"] = accuracy_score(
        y_true=cur_df[both_true_col], y_pred=cur_df[both_pred_col],
    )
    perf_df = pd.DataFrame([perf_df])
    return perf_df