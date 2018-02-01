import torch
import torch.nn as nn

def set_accuracy(y_true, y_pred):
    batch_size, set_size = y_true.size()
    y_pred = (nn.Sigmoid()(y_pred) > 0.5).float()
    correct_classifications = torch.eq(y_pred, y_true)
    set_acc = torch.sum(correct_classifications.sum(dim=1) == set_size).float()/batch_size
    # print(correct_classifications)
    element_acc = torch.sum(correct_classifications).float()/(batch_size*set_size)
    return set_acc, element_acc

def random_samples(batch_size, set_size):
    return torch.rand(batch_size, set_size) > 0.5