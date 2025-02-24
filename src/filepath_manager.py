from os.path import join

from src.misc import create_folder

class FilePath:
    def __init__(self, model_name, fp_checkpoint_folder):
        self.model_name = model_name

        # Folder paths
        self.fp_checkpoint_folder = fp_checkpoint_folder
        self.fp_hp_folder = join(
            self.fp_checkpoint_folder, "hyperparameters", self.model_name)
        self.fp_model_folder = join(
            self.fp_checkpoint_folder, "models", self.model_name)
        self.fp_prediction_folder = join(
            self.fp_checkpoint_folder, "predictions")
        self.fp_performance_folder = join(
            self.fp_checkpoint_folder, "performances")
        self.fp_loss_folder = join(
            self.fp_checkpoint_folder, "losses")
        self.fp_recon_fig_folder = join(
            self.fp_checkpoint_folder, "recon_figs", self.model_name)
        
        # File paths
        self.fp_model_file = join(self.fp_model_folder, "model.pt")
        self.fp_history_file = join(self.fp_model_folder, "history.jpg")
        self.fp_prediction_file = join(self.fp_prediction_folder, self.model_name+".csv")
        self.fp_performance_file = join(self.fp_performance_folder, self.model_name+".csv")
        self.fp_loss_file = join(self.fp_loss_folder, self.model_name+".joblib")
        self.fp_best_recon_fig_file = join(self.fp_recon_fig_folder, "best_recon.jpg")
        self.fp_worst_recon_fig_file = join(self.fp_recon_fig_folder, "worst_recon.jpg")
        
        self.init_folders()
        
    def init_folders(self):
        create_folder(self.fp_hp_folder)
        create_folder(self.fp_model_folder)
        create_folder(self.fp_prediction_folder)
        create_folder(self.fp_performance_folder)
        create_folder(self.fp_loss_folder)
        create_folder(self.fp_recon_fig_folder)
        
        