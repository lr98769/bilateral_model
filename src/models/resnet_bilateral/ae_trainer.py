import torch
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm.auto import tqdm
import time

from src.pytorch_training.misc import freeze_layers_until, count_parameters, count_trainable_parameters
from src.configs.image_config import device 

from src.pytorch_training.callbacks import HistoryRecorder, EarlyStoppingCallback
from src.pytorch_training.display_progress import print_split_epoch_metrics, plot_history
from src.pytorch_training.misc import set_seed_pytorch
from src.models.resnet_bilateral.resnet_bilateral_model import ResNet18Bilateral
from src.models.resnet_bilateral.ae_evaluator import evaluate_w_ae

def train_resnet_ae(
    model: ResNet18Bilateral, 
    train_dl, valid_dl, test_dl, seed,
    fp_model, fp_history=None, # Where to store trained model and history of training
    max_epochs=500, lr=0.001, weight_decay=0.1,  # Training parameters
    patience=5, metric_to_monitor = "mse", maximise=False, # For early stopping
    verbose=True, 
):
    set_seed_pytorch(seed)
    
    metric_list = ["mse"]
    split_list = ["train", "valid"]
    
    # Set Model to Training
    model.train()
    model.to(device)
    
    # Freeze weights before layer 2 of encoder
    freeze_layers_until(model.encoder, layer_name="layer2")
    
    # Set up loss and optimizer
    loss_fn = MSELoss() 
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Check frozen parameters
    print(f"- {count_trainable_parameters(model)} out of {count_parameters(model)} parameters are trainable.")

    # Initialise history recorder, earlystopping callback, metric calculator
    hr = HistoryRecorder(split_list, metric_list)
    es = EarlyStoppingCallback(patience, metric_to_monitor, fp_model, maximise=maximise)
    
    with tqdm(range(max_epochs), leave=False, disable=verbose) as epoch_pbar:
        for epoch in epoch_pbar:
            if epoch > 0:
                epoch_pbar.set_description(f"Valid {metric_to_monitor.capitalize()}: {val_score:.5f}")
            start = time.time()
            all_loss = 0
            total = 0
            position = 0 if verbose else 1 
            with tqdm(train_dl, leave=False, position=position, disable=not verbose) as pbar:
                for input_batch in pbar:
                    input_batch = input_batch.to(device)
                    y_logits = model.reconstruction(input_batch) 
                    
                    # Backprop
                    loss = loss_fn(y_logits, input_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    
                    # Store Loss
                    all_loss+=loss.item()*len(y_logits)
                    total+=len(y_logits)
                    
                    input_batch = input_batch.detach().cpu()
                    loss = loss.detach().cpu()
                    del input_batch, loss
    
            # Calculate train metrics
            train_metric_dict = {"mse":all_loss/total}
    
            # Calculate validation metrics
            valid_metric_dict = {"mse":evaluate_w_ae(model, valid_dl, verbose)}
    
            # Print epoch metrics
            epoch_time = time.time() - start
            if verbose:
                print(f"=== Epoch {epoch}, Time Taken: {epoch_time:.1f}s, Time Left: {epoch_time*(max_epochs-epoch-1):.1f}s ===")
                print_split_epoch_metrics(splitname="train", metric_dict=train_metric_dict)
                print_split_epoch_metrics(splitname="valid", metric_dict=valid_metric_dict)
    
            # Save metrics to visualise training history
            hr.record_epoch_metrics(splitname="train", metric_dict=train_metric_dict)
            hr.record_epoch_metrics(splitname="valid", metric_dict=valid_metric_dict)
    
            # Early Stopping
            val_score = valid_metric_dict[metric_to_monitor]
            if es.stop_training(val_score, epoch, model):
                break
            
    history = hr.get_history_dict()
    plot_history(history, split_list, metric_list, max_cols = 3, fp_history=fp_history)
    return history
    