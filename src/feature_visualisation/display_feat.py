# Visualise Tsne Features
from tqdm.auto import tqdm
from math import ceil
import matplotlib.pyplot as plt

def display_features(feat_dict, col_info, ncols, row_size=2, col_size=2.5, dpi=300, ignore_feat=[], add_legend=False):
    intermediate_classes = list(col_info['intermediate_col_dict'].keys())
    def change_label(label):
        if "IC" in label:
            ic_num = int(label.split(" ")[1])
            return intermediate_classes[ic_num][:-1]+ " IC Feature"
        else:
            return label
    def process_label(label):
        return " ".join(label.split("_"))
    label_col = "label"
    num_classes = len(col_info["output_cols"])
    num_feats = len([feat_label for feat_label in feat_dict if feat_label not in ignore_feat])
    nrows = ceil(num_feats/ncols) 
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*col_size, nrows*row_size), dpi=dpi)
    axes = axes.flatten()
    i = 0
    for (feat_label, feat_df) in tqdm(feat_dict.items(), total=num_feats):
        if feat_label in ignore_feat:
            continue
        ax = axes[i]
        feat_cols = [col for col in feat_df if col != label_col]
        for label, class_df in feat_df.groupby(label_col):
            label = process_label(label)
            ax.scatter(
                class_df[feat_cols[0]], class_df[feat_cols[1]], label=label if i == 0 else None,
                alpha=0.7
            )
        feat_label = process_label(feat_label)
        feat_label = change_label(feat_label)
        ax.set_title(feat_label)
        ax.set_xlabel(process_label(feat_cols[0]))
        if i==0:
            ax.set_ylabel(process_label(feat_cols[1]))
        i+=1
        
    for j in range(i,ncols*nrows):
        axes[j].axis('off')
    if add_legend:
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=num_classes)
    plt.tight_layout()
    