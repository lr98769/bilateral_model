from os.path import join
import torch

# Filepaths
fp_project_folder = join("../../../")
fp_checkpoint_folder = join(fp_project_folder, f"checkpoints") 
fp_data_folder = join(fp_project_folder, "data", "actual")
fp_actual_data_file = join(fp_data_folder, "eye_data.csv")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset column information
input_cols = [
    "Vision Test", "VA", "IOP", "Gradable", "Cup Disc Ratio",
    # OCT
    "OCT RNFL_Abnormal (Progressing)", "OCT RNFL_Abnormal (Stable)",
    "OCT RNFL_Normal", "OCT RNFL_Unreliable", "OCT RNFL_nan",
    # MAC GCA
    "MAC GCA_Abnormal (Progressing)", "MAC GCA_Abnormal (Stable)",
    "MAC GCA_Normal", "MAC GCA_Unreliable", "MAC GCA_nan",
    # HVF
    "HVF_Abnormal (Stable)", "HVF_Normal", "HVF: flat, no IRF/SRF.",
    "HVF_Unreliable", "HVF_nan", "DRF_DH/MA", "DRF_CWS", "DRF_BH",
    # DRF
    "DRF_FH", "DRF_NVE", "DRF_IRMA", "DRF_PRH", "DRF_10M",
    "DMF_DH/MA", "DMF_BH", "DMF_Inner", "DMF_Better", "DMF_HE", 
    # DMF
    "AMDF_DDin", "AMDF_GT125", "AMDF_PA", 
    "AMDF_GA", "AMDF_PED", "AMDF_SFS", "AMDF_SR/subRPE", "AMDF_CNVM", 
    # GSF
    "GSF_RT", "GSF_Notch", "GSF_CDR", "GSF_DA", "GSF_DH",
    # OCTM
    "OCTM_IRF", "OCTM_Normal", "OCTM_Atrophy", "OCTM_ERMpreservedFC",
    "OCTM_Others", "OCTM_ISOSloss", "OCTM_VRtraction", "OCTM_Drusen", 
    "OCTM_ERMdetVA", "OCTM_ERMlossFC", "OCTM_SRF", "OCTM_Ungradable",
    "OCTM_Lamellar", "OCTM_IRHM"
]
intermediate_col_dict = {
    "DMC": ['M0', 'M1', 'M2', 'M3', 'M4', 'NA'],
    "AMDC": ['No','Early','Intermediate','Advanced_1','Advanced_2', "NA"],
    "DRC": ['NoDR', 'MildNPDR', 'ModerateNPDR', 'SevereNPDR', 'ProliferativeDR', 'Unreadable', 'NA'],
    "GSC": ['G0', 'G1', 'NA']
}
tcu_col = 'Total_Time' # Check that this is a string
le_label, re_label = "LE", "RE"
num_input_cols = len(input_cols)