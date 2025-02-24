import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm.auto import tqdm
import time

from src.pytorch_training.misc import freeze_layers_until, count_parameters, count_trainable_parameters
from src.configs.image_config import device
from src.pytorch_training.callbacks import HistoryRecorder, EarlyStoppingCallback, MetricCalculator
from src.pytorch_training.display_progress import print_split_epoch_metrics, plot_history
from src.models.resnet.resnet_inference import predict_w_resnet
from src.pytorch_training.misc import set_seed_pytorch

def train_resnet(
    model, train_dl, valid_dl, test_dl, seed,
    fp_model, fp_history=None, # Where to store trained model and history of training
    max_epochs=500, lr=0.001, weight_decay=0.1,  # Training parameters
    patience=5, metric_to_monitor = "acc", maximise=True, # For early stopping
    verbose=True, 
):
    set_seed_pytorch(seed)
    
    metric_list = ["acc"]
    split_list = ["train", "valid"]
    
    # Set Model to Training
    model.train()
    model.to(device)
    
    # Freeze weights before layer 2
    freeze_layers_until(model, layer_name="layer2")
    
    # Set up loss and optimizer
    loss_fn = CrossEntropyLoss() 
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Check frozen parameters
    print(f"- {count_trainable_parameters(model)} out of {count_parameters(model)} parameters are trainable.")

    # Initialise history recorder, earlystopping callback, metric calculator
    hr = HistoryRecorder(split_list, metric_list)
    es = EarlyStoppingCallback(patience, metric_to_monitor, fp_model, maximise=maximise)
    mc = MetricCalculator(metric_list)
    
    with tqdm(range(max_epochs), leave=False, disable=verbose) as epoch_pbar:
        for epoch in epoch_pbar:
            if epoch > 0:
                epoch_pbar.set_description(f"Valid {metric_to_monitor.capitalize()}: {val_score:.5f}")
            start = time.time()
            all_outputs = None
            position = 0 if verbose else 1 
            with tqdm(train_dl, leave=False, position=position, disable=not verbose) as pbar:
                for input_batch, output_batch in pbar:
                    input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                    y_logits = model(input_batch) 
                    
                    # Backprop
                    loss = loss_fn(y_logits, output_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Store Outputs
                    cur_output_batch = [output_batch, y_logits]
                    cur_output_batch = [element.detach().cpu() for element in cur_output_batch]
                    num_outputs = len(cur_output_batch)
                    
                    if all_outputs is None: # First Batch
                        all_outputs = cur_output_batch
                    else:
                        for i in range(num_outputs):
                            all_outputs[i] = torch.concat((all_outputs[i], cur_output_batch[i]))
                    del input_batch, output_batch, loss
    
            # Calculate train metrics
            train_metric_dict = mc.calculate_metric_dict(
                y_true_label=all_outputs[0], 
                y_pred_label=torch.argmax(all_outputs[1], dim=1))
    
            # Calculate validation metrics
            valid_outputs = predict_w_resnet(model, valid_dl, verbose)
            valid_metric_dict = mc.calculate_metric_dict(
                y_true_label=valid_outputs[0], 
                y_pred_label=torch.argmax(valid_outputs[1], dim=1))
    
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
    