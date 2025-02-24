from os.path import join
import torch

fp_data_folder = "../../../data"
fp_dataset_folder = join(fp_data_folder, "diabetic-retinopathy-unziped")
fp_dataset_csv_file = join(fp_dataset_folder, "train.csv")
fp_data_dfs_file = join(fp_dataset_folder, "data_dfs.joblib")
fp_image_folder = join(fp_dataset_folder, "main train", "main train")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

ori_col_info = {
    "index_col": "image",
    "output_col": "level"
}

col_info = {'id_col': 'id',
 'image_col': 'image',
 'left_label': '_left',
 'right_label': '_right',
 'index_cols': ['id', 'eye'],
 'ordinal_output_cols': ['4>=', '3>=', '2>=', '1>=', '0>='],
 'output_col': 'level',
 "num_classes": 5
}