import numpy as np
import torch
from torch.nn.functional import cross_entropy, softmax,  l1_loss,  mse_loss
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr
from torch.nn.functional import binary_cross_entropy


# Classification Performance Metrics
def get_accuracy(y_label, y_pred):
    return torch.mean((y_pred==y_label).float()).item()

def get_f1(y_label, y_pred):
    return f1_score(y_label, y_pred, average="micro")

def get_mse(y_label, y_pred):
    return torch.mean(torch.square(y_label-y_pred)).item()

def get_ordinal_accuracy(y_label, y_pred):
    y_pred = y_pred.round()
    return torch.mean((y_pred==y_label).all(dim=1).float()).item()

def get_bce(y_label, y_pred):
    return binary_cross_entropy(y_pred, y_label)
