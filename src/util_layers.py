import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenElements(nn.Module):
    def forward(self, x):
        # size(0) == batch
        # size(1) == elements in set
        return x.view(x.size(0), x.size(1), -1)
