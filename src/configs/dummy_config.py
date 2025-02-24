from os.path import join
import torch

# Filepaths
fp_project_folder = join("../")
fp_checkpoint_folder = join(fp_project_folder, f"testing_checkpoints") 
fp_data_folder = join(fp_project_folder, "data", "testing")
fp_actual_data_file = join(fp_data_folder, "dummy_data.csv") 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset column information
input_cols = ["F1", "F2", "F3"]
intermediate_col_dict = {
    "C1":["Diag1", "Diag2", "Diag3"],
    "C2":["Diag1", "Diag2", "Diag3"]
}
tcu_col = "TCU"
le_label, re_label = "LE", "RE"
num_input_cols = len(input_cols)