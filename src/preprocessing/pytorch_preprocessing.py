import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from os.path import join
from torchvision.models import ResNet18_Weights

from src.configs.image_config import fp_image_folder
from src.preprocessing.tabular_preprocessing import get_le_re_feat_cols


class DatasetFromDF(Dataset):
    def __init__(self, df, col_info):
        self.data_df = df
        self.le_cols, self.re_cols = get_le_re_feat_cols(
            le_label=col_info["le_label"], re_label=col_info["re_label"],
            input_cols=col_info["input_cols"])
        self.output_cols = col_info["output_cols"]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        le_features = torch.from_numpy(row[self.le_cols].values).float()
        re_features = torch.from_numpy(row[self.re_cols].values).float()
        label = torch.from_numpy(row[self.output_cols].values).float()
        return (le_features, re_features), label
    
def patient_df_to_eye_df_baseline(df, col_info):
    output_col = col_info["output_col"]
    image_col = col_info["image_col"]
    left_label = col_info["left_label"]
    right_label = col_info["right_label"]
    cols = [image_col, output_col]
    left_cols = [col+left_label for col in cols]
    right_cols = [col+right_label for col in cols]
    new_df = pd.DataFrame(
        np.concatenate((df[left_cols].values, df[right_cols].values), axis=0), 
        columns=cols
    )
    return new_df

def patient_df_to_eye_df_bilateral(df, col_info):
    output_cols = col_info["ordinal_output_cols"]
    image_col = col_info["image_col"]
    left_label = col_info["left_label"]
    right_label = col_info["right_label"]
    cols = [image_col]+output_cols
    left_cols = [col+left_label for col in cols]
    right_cols = [col+right_label for col in cols]
    new_df = pd.DataFrame(
        np.concatenate((df[left_cols].values, df[right_cols].values), axis=0), 
        columns=cols
    )
    return new_df

class BaselineImageDataset(Dataset):
    def __init__(self, df, col_info, transform):
        # self.col_info = col_info
        self.output_col = col_info["output_col"]
        self.image_col = col_info["image_col"]
        self.df = patient_df_to_eye_df_baseline(df, col_info)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = join(fp_image_folder, row[self.image_col]+".jpeg")
        image = self.transform(read_image(img_path))
        label = row[self.output_col]
        return image, label
    
class AEImageDataset(Dataset):
    def __init__(self, df, col_info, transform):
        self.output_col = col_info["ordinal_output_cols"] # Main change
        self.image_col = col_info["image_col"]
        self.df = patient_df_to_eye_df_bilateral(df, col_info)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = join(fp_image_folder, row[self.image_col]+".jpeg")
        image = self.transform(read_image(img_path))
        return image
    
class ICImageDataset(Dataset):
    def __init__(self, df, col_info, transform):
        self.output_col = col_info["ordinal_output_cols"] # Main change
        self.image_col = col_info["image_col"]
        self.df = patient_df_to_eye_df_bilateral(df, col_info)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = join(fp_image_folder, row[self.image_col]+".jpeg")
        image = self.transform(read_image(img_path))
        label = torch.from_numpy(row[self.output_col].values.astype(float)).float()
        return image, label
    
class BCImageDataset(Dataset):
    def __init__(self, df, col_info, transform):
        self.output_cols = [
            output_col+"_"+col_info["id_col"] 
            for output_col in col_info["ordinal_output_cols"]] # Main change
        self.image_col = col_info["image_col"]
        left_label, right_label = col_info["left_label"], col_info["right_label"]
        self.left_image_col = self.image_col+left_label
        self.right_image_col = self.image_col+right_label
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        left_img_path = join(fp_image_folder, row[self.left_image_col]+".jpeg")
        right_img_path = join(fp_image_folder, row[self.right_image_col]+".jpeg")
        left_image = self.transform(read_image(left_img_path))
        right_image = self.transform(read_image(right_img_path))
        label = torch.from_numpy(row[self.output_cols].values.astype(float)).float()
        return (left_image, right_image), label
        
def get_dl(df, col_info, batch_size, shuffle):
    ds = DatasetFromDF(df=df, col_info=col_info)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

def get_baseline_image_dl(df, col_info, batch_size, shuffle):
    ds = BaselineImageDataset(df, col_info, transform=ResNet18_Weights.IMAGENET1K_V1.transforms())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

def get_ae_image_dl(df, col_info, batch_size, shuffle):
    ds = AEImageDataset(df, col_info, transform=ResNet18_Weights.IMAGENET1K_V1.transforms())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

def get_ic_image_dl(df, col_info, batch_size, shuffle):
    ds = ICImageDataset(df, col_info, transform=ResNet18_Weights.IMAGENET1K_V1.transforms())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

def get_bc_image_dl(df, col_info, batch_size, shuffle):
    ds = BCImageDataset(df, col_info, transform=ResNet18_Weights.IMAGENET1K_V1.transforms())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl
    
def get_pytorch_split_dict(data_dict, col_info, batch_size, eval_batch_size, shuffle_train=True, dl_func=get_dl):
    dl_dict = {}
    for split_name in data_dict.keys():
        cur_batch_size = batch_size if split_name == "train_df" else eval_batch_size
        cur_shuffle = shuffle_train if split_name == "train_df" else False
        dl_label = split_name[:-1] + "l" # 'train_df' -> 'train_dl'
        df = data_dict[split_name]
        dl_dict[dl_label] = dl_func(
            df=df, col_info=col_info, batch_size=cur_batch_size, shuffle=cur_shuffle)
    return dl_dict

