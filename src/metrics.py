import torch
import torch.nn as nn

def set_accuracy(y_true, y_pred):
    """
    Calculates the accuracy of selection based on
    ground truth selection idx
    :param y_true: Ground Truth Selection IDX
    :param y_pred: Predicted Selection IDX
    :return:
    """
    if len(tuple(y_true.size())) == 3:
        y_true = y_true.squeeze(-1).float()
    if len(tuple(y_pred.size())) == 3:
        y_pred = y_pred.squeeze(-1).float()
    batch_size, set_size = y_true.size()
    y_pred = (nn.Sigmoid()(y_pred) > 0.5).float()
    correct_classifications = (y_true == y_pred).float()
    elem_acc = torch.mean(correct_classifications)
    set_acc = (torch.sum(correct_classifications, 1) == set_size).float().mean()
    return set_acc, elem_acc

def random_samples(batch_size, set_size):
    return torch.rand(batch_size, set_size) > 0.5
