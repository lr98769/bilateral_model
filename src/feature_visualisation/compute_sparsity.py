import numpy as np
import pandas as pd

def calculate_sparsity(matrix: np.array, dp=5):
    assert len(matrix.shape) == 2 # Ensure that this matrix is 2d
    rounded_matrix = np.round(matrix, decimals=dp)
    num_non_zero = np.sum(rounded_matrix>0)
    total_num_elements = rounded_matrix.shape[0] * rounded_matrix.shape[1]
    sparsity = 1 - num_non_zero/total_num_elements
    return sparsity

def calculate_sparsity_of_encoder_features(encoder_feat_dict:dict, dp=5):
    output_df = []
    both_eye_feats = []
    for eye_label, eye_feat in encoder_feat_dict.items():
        if eye_label == "label":
            continue
        eye_feat = np.array(eye_feat)
        output_df.append({
            "Eye": eye_label.split("_")[0].upper(),
            "Sparsity": calculate_sparsity(matrix=eye_feat, dp=dp)
        })
        both_eye_feats.append(eye_feat)
    both_eye_feats = np.concatenate(both_eye_feats, axis=0)
    output_df.append({
        "Eye": "Both",
        "Sparsity": calculate_sparsity(matrix=both_eye_feats, dp=dp)
    })
    return pd.DataFrame(output_df)
       