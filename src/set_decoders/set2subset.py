import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from src.set_encoders import ContextFreeEncoder


class SimpleSubset(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=10,
                 prob_net_layers=1,
                 logprobs=False):
        super().__init__()
        assert prob_net_layers >= 1, 'Must have at least one linear layer'

        if prob_net_layers == 1:
            hidden_dim = 1

        seq = nn.Sequential(
                nn.Linear(input_dim, hidden_dim)
            )

        for l in range(prob_net_layers-1):
            seq.add_module(str(l)+'relu', nn.ReLU())
            seq.add_module(str(l)+'linear', nn.Linear(hidden_dim, hidden_dim))
        
        if not logprobs:
            seq.add_module('sigmoid', nn.Sigmoid())
        
        self.prob = ContextFreeEncoder(seq, '1d')

    def forward(self, x):
        x = self.prob(x)
        return x
        