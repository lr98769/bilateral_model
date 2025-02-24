from src.misc import set_seed
import numpy as np
import pandas as pd

def get_le_re_feat_cols(le_label, re_label, input_cols):
    le_cols = [feat_col + f" {le_label}" for feat_col in input_cols]
    re_cols = [feat_col + f" {re_label}" for feat_col in input_cols]
    return le_cols, re_cols

def preprocess_data_for_baseline(df, input_cols, output_cols, le_label, re_label):
    df = df.copy()
    le_cols, re_cols = get_le_re_feat_cols(le_label, re_label, input_cols)
    return df[le_cols+re_cols].values, df[output_cols].values

def preprocess_data_for_ae(df, input_cols, le_label, re_label):
    df = df.copy()
    le_cols, re_cols = get_le_re_feat_cols(le_label, re_label, input_cols)
    return np.concatenate([df[le_cols].values, df[re_cols].values], axis=0)

def preprocess_data_for_ic(df, input_cols, intermediate_cols, intermediate_col_sizes, le_label, re_label):
    df = df.copy()
    # Get Input
    le_input_cols, re_input_cols = get_le_re_feat_cols(le_label, re_label, input_cols)
    le_ic_cols, re_ic_cols = get_le_re_feat_cols(le_label, re_label, intermediate_cols)
    inputs = np.concatenate([df[le_input_cols].values, df[re_input_cols].values], axis=0)
    
    # Get Output
    start = 0
    outputs = []
    for size in intermediate_col_sizes:
        cur_le_ic_cols, cur_re_ic_cols = le_ic_cols[start:start+size], re_ic_cols[start:start+size]
        cur_output = np.concatenate(
            [df[cur_le_ic_cols].values, df[cur_re_ic_cols].values], axis=0)
        outputs.append(cur_output)
        start += size
    return inputs, outputs

def preprocess_data_for_fc(df, input_cols, output_cols, le_label, re_label):
    df = df.copy()
    le_cols, re_cols = get_le_re_feat_cols(le_label, re_label, input_cols)
    return (df[le_cols].values, df[re_cols].values), df[output_cols].values

def preprocess_data_for_ae_ic_fc(df, input_cols, intermediate_cols, output_cols, le_label, re_label):
    df = df.copy()
    le_input_cols, re_input_cols = get_le_re_feat_cols(le_label, re_label, input_cols)
    le_intermediate_cols, re_intermediate_cols = get_le_re_feat_cols(le_label, re_label, intermediate_cols)
    return (
        [df[le_input_cols].values, df[re_input_cols].values], 
        [df[le_intermediate_cols].values, df[re_intermediate_cols].values], 
        df[output_cols].values
    )
    
def generate_le_re_df(le_mat, re_mat, col_list, le_label, re_label, suffix, df):
    mat = np.stack((le_mat, re_mat))
    # Assume mat shape = 2, num_samples, features
    # Transpose to: (num_samples, 2, num features)
    mat = np.transpose(mat, (1, 0, 2))
    # Reshape to: (num_samples, 2 * num features)
    mat = mat.reshape(len(mat), -1)
    colnames = [f"{col} {le_label}_{suffix}" for col in col_list] + [f"{col} {re_label}_{suffix}" for col in col_list]
    return pd.DataFrame(mat, columns=colnames, index=df.index)

def preprocess_data_for_scikitlearn(df, input_cols, output_cols, le_label, re_label):
    df = df.copy()
    le_cols, re_cols = get_le_re_feat_cols(le_label, re_label, input_cols)
    Y = df[output_cols].values
    y = np.argmax(Y, axis=-1)
    return df[le_cols+re_cols].values, Y, y

def process_data_for_feat_visualisation(pred_df, col_info):
    (X_le, X_re), Y = preprocess_data_for_fc(
        pred_df, 
        input_cols=col_info["input_cols"], 
        output_cols=col_info["output_cols"], 
        le_label=col_info["le_label"], re_label=col_info["re_label"]
    )
    Y_label = Y.argmax(axis=1)
    Y_label = [col_info["output_cols"][label] for label in Y_label]
    return (X_le, X_re), Y_label

def add_noise(data_dfs, std, col_info, seed):
    set_seed(seed)
    new_data_dfs = {}
    le_cols, re_cols = get_le_re_feat_cols(
        le_label=col_info["le_label"], re_label=col_info["re_label"], 
        input_cols=col_info["input_cols"])
    feat_cols = le_cols + re_cols
    num_feat = len(feat_cols)
    for label, df in data_dfs.items():
        df = df.copy()
        noise = np.random.normal(scale=std, size=[len(df), num_feat]) 
        df[feat_cols] += noise
        new_data_dfs[label] = df
    return new_data_dfs

def choose_output_cols(col_info, bilateral=True):
    col_info = col_info.copy()
    if bilateral:
        col_info.pop("output_cols_baseline")
        output_cols = col_info.pop("output_cols_bilateral")
    else:
        col_info.pop("output_cols_bilateral")
        output_cols = col_info.pop("output_cols_baseline")
    col_info["output_cols"] =  output_cols
    return col_info

