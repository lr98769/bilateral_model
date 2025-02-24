import numpy as np



def get_best_val_loss(history, metric):
    best_epoch = np.argmin(history.history["val_loss"])
    best_val_loss = min(history.history["val_loss"])
    best_val_loss_wo_sparsity = history.history[f"val_{metric}"][best_epoch]
    return best_val_loss, best_val_loss_wo_sparsity, best_epoch

def transfer_ae(base_bilateral_model, new_bilateral_model):
    new_bilateral_model.encoder = base_bilateral_model.encoder
    new_bilateral_model.decoder = base_bilateral_model.decoder
    return new_bilateral_model

def transfer_ae_n_ic(base_bilateral_model, new_bilateral_model):
    new_bilateral_model = transfer_ae(base_bilateral_model, new_bilateral_model)
    new_bilateral_model.intermediate_classifier = base_bilateral_model.intermediate_classifier
    return new_bilateral_model


