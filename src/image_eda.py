import numpy as np
import torch
import matplotlib.pyplot as plt
from math import ceil
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from src.label_converter import ImageLabelConverter

def get_class_distribution(data_dfs, col_info):
    output_col = col_info["output_col"]
    id_col = col_info["id_col"]
    col = output_col+"_"+id_col
    class_dist_df = []
    index = []
    for df_label, df in data_dfs.items():
        class_dist_df.append(df[col].value_counts().to_dict())
        index.append(df_label.split("_")[0].capitalize())
    class_dist_df = pd.DataFrame(class_dist_df, index=index)
    class_dist_df = class_dist_df[class_dist_df.columns.sort_values()]
    return class_dist_df

def transform_for_visualisation(img: torch.tensor, data_type=np.float32, permute=True, reverse_transform=True):
    if permute:
        img = img.permute((1, 2, 0)).numpy()
    img_copy = img.copy()
    img_copy = img_copy.astype(data_type) 
    if reverse_transform:
        img_copy[:, :, 0] = img_copy[:, :, 0]* 0.229 + 0.485
        img_copy[:, :, 1] = img_copy[:, :, 1]* 0.224 + 0.456
        img_copy[:, :, 2] = img_copy[:, :, 2]*0.225 + 0.406
    return (img_copy.clip(0, 1) * 255).astype(np.uint8)

def show_img_examples(ds_or_dl, col_info, ncols=1, img_size=2, dpi=300):
    if isinstance(ds_or_dl, Dataset):
        show_img_examples_ds(ds_or_dl, col_info, ncols, img_size, dpi)
    else:
        show_img_examples_dl(ds_or_dl, col_info, ncols, img_size, dpi)

def show_img_examples_ds(ds: Dataset, col_info, ncols=1, img_size=2, dpi=300):
    num_classes = col_info["num_classes"]
    displayed_class_labels = []
    nrows = ceil(num_classes/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*ncols, img_size*nrows), dpi=dpi)
    axes = axes.flatten()
    for img, label in ds:
        if label not in displayed_class_labels:
            axes[label].imshow(transform_for_visualisation(img))
            axes[label].set_title(f"Class {label}")
            displayed_class_labels.append(label)
        if len(displayed_class_labels) >= num_classes:
            break
    plt.tight_layout()
    plt.show()
    
def show_img_examples_dl(dl: DataLoader, col_info, ncols=1, img_size=2, dpi=300):
    num_classes = col_info["num_classes"]
    displayed_class_labels = []
    nrows = ceil(num_classes/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*ncols, img_size*nrows), dpi=dpi)
    axes = axes.flatten()
    for img_batch, label_batch in dl:
        for i in range(len(img_batch)):
            img, label = img_batch[i], label_batch[i]
            if label not in displayed_class_labels:
                axes[label].imshow(transform_for_visualisation(img))
                axes[label].set_title(f"Class {label}")
                displayed_class_labels.append(label)
            if len(displayed_class_labels) >= num_classes:
                break
        if len(displayed_class_labels) >= num_classes:
                break
    plt.tight_layout()
    plt.show()
    
def show_ae_img_examples(ds_or_dl, col_info, num_examples=4, ncols=4, img_size=2, dpi=300):
    if isinstance(ds_or_dl, Dataset):
        show_ae_img_examples_ds(ds_or_dl, col_info, num_examples, ncols, img_size, dpi)
    else:
        show_ae_img_examples_dl(ds_or_dl, col_info, num_examples, ncols, img_size, dpi)
        
def show_ae_img_examples_ds(ds: Dataset, col_info, num_examples, ncols=4, img_size=2, dpi=300):
    nrows = ceil(num_examples/ncols)
    displayed_examples = 0
    print(nrows, ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*nrows, img_size*ncols), dpi=dpi)
    axes = axes.flatten()
    for img in ds:
        axes[displayed_examples].imshow(transform_for_visualisation(img))
        displayed_examples+=1
        if displayed_examples >= num_examples:
            break
    plt.tight_layout()
    plt.show()
    
def show_ae_img_examples_dl(dl: DataLoader, col_info, num_examples, ncols=4, img_size=2, dpi=300):
    nrows = ceil(num_examples/ncols)
    displayed_examples = 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*nrows, img_size*ncols), dpi=dpi)
    axes = axes.flatten()
    for img_batch in dl:
        for i in range(len(img_batch)):
            img = img_batch[i]
            axes[displayed_examples].imshow(transform_for_visualisation(img))
            displayed_examples+=1
            if displayed_examples >= num_examples:
                break
        if displayed_examples >= num_examples:
            break
    plt.tight_layout()
    plt.show()
    
def show_ic_img_examples(ds_or_dl, col_info, ncols=5, img_size=2, dpi=300):
    if isinstance(ds_or_dl, Dataset):
        show_ic_img_examples_ds(ds_or_dl, col_info, ncols, img_size, dpi)
    else:
        show_ic_img_examples_dl(ds_or_dl, col_info, ncols, img_size, dpi)

def show_ic_img_examples_ds(ds: Dataset, col_info, ncols=1, img_size=2, dpi=300):
    num_classes = col_info["num_classes"]
    label_converter = ImageLabelConverter(num_classes)
    displayed_class_labels = []
    nrows = ceil(num_classes/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*ncols, img_size*nrows), dpi=dpi)
    axes = axes.flatten()
    for img, label in ds:
        label_converted = label_converter.decode_label(label)
        if label_converted not in displayed_class_labels:
            axes[label_converted].imshow(transform_for_visualisation(img))
            axes[label_converted].set_title(f"Class {label.numpy()}")
            displayed_class_labels.append(label_converted)
        if len(displayed_class_labels) >= num_classes:
            break
    plt.tight_layout()
    plt.show()
    
def show_ic_img_examples_dl(dl: DataLoader, col_info, ncols=5, img_size=2, dpi=300):
    num_classes = col_info["num_classes"]
    label_converter = ImageLabelConverter(num_classes)
    displayed_class_labels = []
    nrows = ceil(num_classes/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*ncols, img_size*nrows), dpi=dpi)
    axes = axes.flatten()
    for img_batch, label_batch in dl:
        for i in range(len(img_batch)):
            img, label = img_batch[i], label_batch[i]
            label_converted = label_converter.decode_label(label)
            if label_converted not in displayed_class_labels:
                axes[label_converted].imshow(transform_for_visualisation(img))
                axes[label_converted].set_title(f"Class {label.numpy()}")
                displayed_class_labels.append(label_converted)
            if len(displayed_class_labels) >= num_classes:
                break
        if len(displayed_class_labels) >= num_classes:
                break
    plt.tight_layout()
    plt.show()
    
def show_bc_img_examples(ds_or_dl, col_info, img_size=2, dpi=300):
    if isinstance(ds_or_dl, Dataset):
        show_bc_img_examples_ds(ds_or_dl, col_info, img_size, dpi)
    else:
        show_bc_img_examples_dl(ds_or_dl, col_info, img_size, dpi)

def show_bc_img_examples_ds(ds: Dataset, col_info, img_size=2, dpi=300):
    num_classes = col_info["num_classes"]
    label_converter = ImageLabelConverter(num_classes)
    displayed_class_labels = []
    nrows, ncols = 2, num_classes
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*ncols, img_size*nrows), dpi=dpi)
    for (left_img, right_img), label in ds:
        label_converted = label_converter.decode_label(label)
        if label_converted not in displayed_class_labels:
            axes[0, label_converted].set_title(f"Class {label.numpy()}")
            axes[0, label_converted].imshow(transform_for_visualisation(left_img))
            axes[1, label_converted].imshow(transform_for_visualisation(right_img))
            displayed_class_labels.append(label_converted)
        if len(displayed_class_labels) >= num_classes:
            break
    axes[0, 0].set_ylabel("Left")
    axes[0, 0].set_ylabel("Right")
    plt.tight_layout()
    plt.show()
    
def show_bc_img_examples_dl(dl: DataLoader, col_info, img_size=2, dpi=300):
    num_classes = col_info["num_classes"]
    label_converter = ImageLabelConverter(num_classes)
    displayed_class_labels = []
    nrows, ncols = 2, num_classes
    fig, axes = plt.subplots(nrows, ncols, figsize=(img_size*ncols, img_size*nrows), dpi=dpi)
    for (left_img_batch, right_img_batch), label_batch in dl:
        for i in range(len(label_batch)):
            left_img, right_img, label = left_img_batch[i], right_img_batch[i], label_batch[i]
            label_converted = label_converter.decode_label(label)
            if label_converted not in displayed_class_labels:
                axes[0, label_converted].set_title(f"Class {label.numpy()}")
                axes[0, label_converted].imshow(transform_for_visualisation(left_img))
                axes[1, label_converted].imshow(transform_for_visualisation(right_img))
                displayed_class_labels.append(label_converted)
            if len(displayed_class_labels) >= num_classes:
                break
        if len(displayed_class_labels) >= num_classes:
                break
    axes[0, 0].set_ylabel("Left")
    axes[0, 0].set_ylabel("Right")
    plt.tight_layout()
    plt.show()
    

