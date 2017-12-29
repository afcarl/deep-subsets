import torch
import torch.nn as nn

def set_accuracy(y_true, y_pred):
    batch_size, set_size = y_true.size()
    y_pred = nn.Sigmoid()(y_pred)
    acc = torch.sum(torch.eq(y_pred, y_true).sum(dim=1) == set_size).float()/batch_size
    return acc

def random_samples(batch_size, set_size):
    return torch.rand(batch_size, set_size) > 0.5