import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class LinearSumSet(nn.Module):
    """
    Returns the linear sum of 
    every element in the set
    """
    def forward(self, x):
        return torch.sum(x, dim=1)