import torch
import numpy as np

from src.pytorch_training.metrics import get_accuracy, get_f1, get_mse, get_ordinal_accuracy, get_bce

class EarlyStoppingCallback:
    def __init__(self, patience, metric_to_monitor, fp_model, maximise=True):
        self.patience = patience
        self.metric_to_monitor = metric_to_monitor
        self.fp_model = fp_model
        self.maximise = maximise
        
        self.best_epoch = -1
        self.best_val_score = -np.inf if self.maximise else np.inf
        self.num_non_improving_epochs = 0

    def improved(self, val_score):
        if self.maximise:
            return val_score > self.best_val_score
        else:
            return val_score < self.best_val_score
        
    def stop_training(self, val_score, epoch, model):
        # Output True if we should stop training
        # Output False if we should continue
        # If there is an improvement
        if self.improved(val_score):
            self.best_val_score = val_score
            self.best_epoch = epoch
            self.num_non_improving_epochs = 0
            torch.save(model, self.fp_model)
            return False
        # No improvement
        else:
            self.num_non_improving_epochs += 1
            # If it hasn't improved in a long time
            if (self.num_non_improving_epochs >= self.patience):
                print(
                    f"Early Stopping at Epoch {epoch}, "
                    f"Best Validation {self.metric_to_monitor.capitalize()} ({self.best_val_score:.5f}) at Epoch {self.best_epoch}.")
                # Stop training
                return True
            else:
                # Continue training
                return False

class HistoryRecorder:
    def __init__(self, split_list, metric_list):
        self.split_list = split_list
        self.metric_list = metric_list
        self.epoch_metrics = dict()
        self.history = dict()
        for split_name in self.split_list:
            self.history[split_name] = dict()
            for metric in self.metric_list:
                self.history[split_name][metric] = []

    def record_epoch_metrics(self, splitname, metric_dict):
        for metric in self.metric_list:
            self.history[splitname][metric].append(metric_dict[metric])

    def get_history_dict(self):
        return self.history

class MetricCalculator:
    def __init__(self, metric_list, beta=None):
        self.metric_list = metric_list
        if beta:
            self.beta = beta
        self.classifier_metric_components_dict = {
            "acc" : get_accuracy, "f1" : get_f1,
        }
        self.ordinal_metric_components_dict = {
            "ordinal_acc": get_ordinal_accuracy, "bce": get_bce
        }
        self.ae_metric_components_dict = {"mse": get_mse}
        self.classifier_metric_components = {}
        self.ae_metric_components = {}
        self.ordinal_metric_components = {}
        for metric in self.metric_list:
            if metric in self.classifier_metric_components_dict:
                self.classifier_metric_components[metric] = self.classifier_metric_components_dict[metric]
            elif metric in self.ae_metric_components_dict:
                self.ae_metric_components[metric] = self.ae_metric_components_dict[metric]
            elif metric in self.ordinal_metric_components_dict:
                self.ordinal_metric_components[metric] = self.ordinal_metric_components_dict[metric]
            else:
                raise Exception(f"{metric} is not a valid metric!")

    def calculate_metric_dict(self, y_true_label, y_pred_label):
        metric_dict = dict()
        # Classifier metrics
        for metric_name, metric_function in self.classifier_metric_components.items():
            metric_dict[metric_name] = metric_function(y_true_label, y_pred_label)
        # AE metrics
        for metric_name, metric_function in self.ae_metric_components.items():
            metric_dict[metric_name] = metric_function(y_true_label, y_pred_label)
        # Ordinal metrics
        for metric_name, metric_function in self.ordinal_metric_components.items():
            metric_dict[metric_name] = metric_function(y_true_label, y_pred_label)
        return metric_dict