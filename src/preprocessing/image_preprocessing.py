import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from numpy import random, arange
from os.path import exists, join
from torchvision.io import read_image

from src.label_converter import ImageLabelConverter
from src.configs.image_config import fp_image_folder


def sort_image_columns(cols):
    def search_list(cur_list, keyword):
        return [item for item in cur_list if keyword in item]
    # id, image_left, image_right, 
    #  level_left, level_right, level_id, 
    #  4>=_left...., 4>=_right..., 4>=_id...
    image_cols = search_list(cols, keyword="image")
    level_cols = search_list(cols, keyword="level")
    ordinal_cols = search_list(cols, keyword=">=")
    return image_cols + level_cols + ordinal_cols


def process_image_df(df: pd.DataFrame, col_info: dict):
    df = df.copy()
    index_col = col_info["index_col"]
    output_col = col_info["output_col"]
    num_classes = df[output_col].nunique()
    num_samples = len(df)

    # 1. Process Index
    # "10_left" -> "10" and "left"
    id_col, eye_col = "id", "eye"
    new_index_cols = [id_col, eye_col]
    df[new_index_cols] = df[index_col].str.split("_", expand=True)

    # 2. Encode into Special Ordinal Style
    # - 4 is the worst state, 0 is the least severe state
    # if LE is 4 and RE is 0, we should output 4
    # 4 = [1 0 0 0, 0], 1 = [1, 1, 1, 1, 1], 4*1 = [1 0 0 0, 0]
    label_converter = ImageLabelConverter(num_classes=num_classes)
    new_label_cols = [f"{i}>=" for i in range(num_classes-1, -1, -1)]
    labels = []
    for i in tqdm(range(num_samples), total=num_samples):
        label = label_converter.encode_class_num(df[output_col].iloc[i])
        labels.append(label)
    df[new_label_cols] = labels

    # id, left_image, right_image, left_level, right_level, id_level, left_4>=...., right_4>=..., id_4>=...
    left_label, right_label = "left", "right"
    left_df = df[df[eye_col]==left_label].set_index(id_col)
    right_df = df[df[eye_col]==right_label].set_index(id_col)
    df = left_df.join(right_df, lsuffix="_"+left_label, rsuffix="_"+right_label, on=id_col)

    # Make patient labels
    patient_label = "_id"
    df[output_col+patient_label] = np.max(
        df[[output_col+"_"+left_label, output_col+"_"+right_label]].values, axis=-1)
    df[[col+patient_label for col in new_label_cols]] = (
        df[[col+"_"+left_label for col in new_label_cols]].values *
        df[[col+"_"+right_label for col in new_label_cols]].values)

    df = df[sort_image_columns(df.columns)]

    col_info = {
        "id_col": id_col,
        "image_col": index_col,
        "left_label": "_"+left_label,
        "right_label": "_"+right_label,
        "index_cols": new_index_cols,
        "ordinal_output_cols": new_label_cols,
        "output_col": output_col,
        "num_classes": num_classes
    }

    return df, col_info

def split_df(processed_df,  valid_prop, test_prop, seed):
    random.seed(seed=seed)
    size = len(processed_df)
    train_prop = 1 - valid_prop - test_prop
    train_size, valid_size = round(size*train_prop), round(size*(train_prop+valid_prop))
    indices = arange(size)
    random.shuffle(indices)
    train_indices, valid_indices, test_indicies = (
        indices[:train_size], indices[train_size:valid_size], indices[valid_size:])
    return {
        "train_df": processed_df.iloc[train_indices], 
        "valid_df": processed_df.iloc[valid_indices], 
        "test_df": processed_df.iloc[test_indicies]}
    

def remove_rows_where_any_image_is_missing_or_truncated(processed_df):
    length_before = len(processed_df)
    indices_to_keep = []
    num_samples = len(processed_df)
    for i in tqdm(range(num_samples), total=num_samples):
        left_img = join(fp_image_folder, processed_df["image_left"].iloc[i] + ".jpeg")
        right_img = join(fp_image_folder, processed_df["image_right"].iloc[i] + ".jpeg")
        if exists(left_img) and exists(right_img):
            try:
                read_image(left_img)
                read_image(right_img)
            except:
                continue
            indices_to_keep.append(i)
    processed_df = processed_df.iloc[indices_to_keep]
    print(f"Before: {length_before}")
    print(f"- Removed: {len(processed_df)-length_before}")
    print(f"After: {len(processed_df)}")
    return processed_df