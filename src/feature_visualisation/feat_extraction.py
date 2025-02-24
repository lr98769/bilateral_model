from keras import backend as K
import time
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA

from src.models.bilateral.get_model import get_intermediate_classifier_model
from src.preprocessing.tabular_preprocessing import process_data_for_feat_visualisation, preprocess_data_for_fc

def get_layer_output(x, input_layer, output_layer):
    func = K.function([input_layer.input],
                [output_layer.output])
    output = func([x])[0]
    return output

def get_encoder_features(model, X_le, X_re, Y_label):
    feat_dict = {}
    input_layer = model.encoder.layers[0]
    output_layer = model.encoder.layers[-1]
    le_enc_feat = get_layer_output(X_le, input_layer, output_layer)
    re_enc_feat = get_layer_output(X_re, input_layer, output_layer)
    feat_dict[f"Encoder_Feature"]= {
        "le_feat": le_enc_feat, "re_feat": re_enc_feat, "label": Y_label}
    return feat_dict

def get_ic_features(model, le_enc_feat, re_enc_feat, Y_label):
    feat_dict = {}
    num_intermediate_classifiers = len(model.intermediate_classifier.layers)
    for i in tqdm(range(num_intermediate_classifiers), total=num_intermediate_classifiers):
        input_layer = model.intermediate_classifier.layers[i].layers[0]
        output_layer = model.intermediate_classifier.layers[i].layers[-2]
        le_feat = get_layer_output(le_enc_feat, input_layer, output_layer)
        re_feat = get_layer_output(re_enc_feat, input_layer, output_layer)
        feat_dict[f"IC_{i}_Feature"] = {
            "le_feat": le_feat, "re_feat": re_feat, "label": Y_label}
    return feat_dict

def get_ic_diagnosis(model, X_le, X_re, Y_label):
    def get_ic(cur_input):
        return model.concatenate(model.intermediate_classifier(model.encoder(cur_input)))
    feat_dict = {}
    X_le_ic = get_ic(X_le)
    X_re_ic = get_ic(X_re)
    feat_dict[f"IC_Diagnosis"]= {
        "le_feat": X_le_ic, "re_feat": X_re_ic, "label": Y_label}
    return feat_dict
    
def get_fc_features(model, X_le, X_re, Y_label):
    feat_dict = {}
    ic_model = get_intermediate_classifier_model(model)
    le_ic_output = model.concatenate(ic_model(X_le))
    re_ic_output = model.concatenate(ic_model(X_re))
    input_layer = model.final_classifier.layers[0]
    output_layer = model.final_classifier.layers[-2]
    le_feat = get_layer_output(le_ic_output, input_layer, output_layer)
    re_feat = get_layer_output(re_ic_output, input_layer, output_layer)
    feat_dict[f"FC_Feature"] = {
        "le_feat": le_feat, "re_feat": re_feat, "label": Y_label}
    return feat_dict
    
def get_all_layer_outputs(model, pred_df, col_info):
    # Store Layer Outputs here
    layer_output_dict = {}
    
    # Preprocess Data
    (pred_X_le, pred_X_re), pred_Y_label= process_data_for_feat_visualisation(
        pred_df=pred_df, col_info=col_info)
    
    # Get After Encoder Features
    print("1. Get Encoder Features")
    start = time.time()
    enc_feat = get_encoder_features(model, pred_X_le, pred_X_re, pred_Y_label)
    layer_output_dict.update(enc_feat)
    print(f"\t - Took {time.time()-start}s")
    
    # Get Output From Each Intermediate Classifier
    print(f"2. Get Intermediate Classifier Features")
    start = time.time()
    ic_feat = get_ic_features(
        model, 
        le_enc_feat=enc_feat[f"Encoder_Feature"]["le_feat"],
        re_enc_feat=enc_feat[f"Encoder_Feature"]["re_feat"],
        Y_label=pred_Y_label
    )
    layer_output_dict.update(ic_feat)
    print(f"\t - Took {time.time()-start}s")
        
    # Get Output From Each Final Classifier
    print(f"3. Get Final Classifier Features")
    start = time.time()
    fc_feat = get_fc_features(
        model, X_le=pred_X_le, X_re=pred_X_re, Y_label=pred_Y_label
    )
    layer_output_dict.update(fc_feat)
    print(f"\t - Took {time.time()-start}s")
        
    return layer_output_dict

def get_tsne_features(layer_output_dict, seed, perplexity=3, n_components = 2):
    tsne_cols = [f"tsne_{i}" for i in range(n_components)]
    tsne_feat_dict = {}
    for feat_label, feat_dict in tqdm(layer_output_dict.items(), total=len(layer_output_dict)):
        le_feat, re_feat, label = feat_dict["le_feat"], feat_dict["re_feat"], feat_dict["label"]
        concatenated_feat = np.concatenate([le_feat, re_feat], axis=1)
        tsne_feat = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed).fit_transform(concatenated_feat)
        tsne_df = pd.DataFrame(tsne_feat, columns=tsne_cols)
        tsne_df["label"] = label
        tsne_feat_dict[feat_label] = tsne_df
    return tsne_feat_dict

def get_pca_features(layer_output_dict, seed, perplexity=3, n_components = 2):
    pca_cols = [f"pca_{i}" for i in range(n_components)]
    pca_feat_dict = {}
    for feat_label, feat_dict in tqdm(layer_output_dict.items(), total=len(layer_output_dict)):
        le_feat, re_feat, label = feat_dict["le_feat"], feat_dict["re_feat"], feat_dict["label"]
        concatenated_feat = np.concatenate([le_feat, re_feat], axis=1)
        pca_feat = PCA(n_components=n_components, random_state=seed).fit_transform(concatenated_feat)
        pca_df = pd.DataFrame(pca_feat, columns=pca_cols)
        pca_df["label"] = label
        pca_feat_dict[feat_label] = pca_df
    return pca_feat_dict

def format_encoder_features_for_training(model, data_dfs, col_info):
    new_data_dfs = {}
    for i, (df_label, df) in enumerate(data_dfs.items()):
        (X_le, X_re), Y = preprocess_data_for_fc(
            df, 
            input_cols=col_info["input_cols"], 
            output_cols=col_info["output_cols"], 
            le_label=col_info["le_label"], re_label=col_info["re_label"]
        )
        # Get After Encoder Features
        enc_feat = get_encoder_features(model, X_le, X_re, Y)["Encoder_Feature"]
        # Get Feature Columns
        if i == 0:
            num_feats = enc_feat["le_feat"].shape[1]
            feat_cols = [f"enc_feat_{i}" for i in range(num_feats)]
        else:
            num_feats = enc_feat["le_feat"].shape[1]
            assert num_feats == len(feat_cols)
        # Generate new df
        matrix = np.concatenate((enc_feat["le_feat"], enc_feat["re_feat"], enc_feat["label"]), axis=1)
        colnames = (
            [feat+" LE" for feat in feat_cols] + 
            [feat+" RE" for feat in feat_cols] + 
            col_info["output_cols"])
        df = pd.DataFrame(matrix, columns=colnames)
        new_data_dfs[df_label] = df
    new_col_info = col_info.copy()
    new_col_info["input_cols"] = feat_cols
    return new_data_dfs, new_col_info

def format_diagnosis_classification_for_training(model, data_dfs, col_info):
    new_data_dfs = {}
    for i, (df_label, df) in enumerate(data_dfs.items()):
        (X_le, X_re), Y = preprocess_data_for_fc(
            df, 
            input_cols=col_info["input_cols"], 
            output_cols=col_info["output_cols"], 
            le_label=col_info["le_label"], re_label=col_info["re_label"]
        )
        # Get After IC Diagnosis
        enc_feat = get_ic_diagnosis(model, X_le, X_re, Y)["IC_Diagnosis"]
        # Get Feature Columns
        feat_cols = model.intermediate_cols.numpy()
        feat_cols = [col.decode("utf-8") for col in feat_cols]
        # Generate new df
        matrix = np.concatenate((enc_feat["le_feat"], enc_feat["re_feat"], enc_feat["label"]), axis=1)
        colnames = (
            [feat+" LE" for feat in feat_cols] + 
            [feat+" RE" for feat in feat_cols] + 
            col_info["output_cols"])
        df = pd.DataFrame(matrix, columns=colnames)
        new_data_dfs[df_label] = df
    new_col_info = col_info.copy()
    new_col_info["input_cols"] = feat_cols
    return new_data_dfs, new_col_info