# encode elements of a set
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class ContextBasedLinear(nn.Module):
    def __init__(self, nonlinearity=None):
        super().__init__()
        self.mu = Parameter(torch.Tensor(1))
        self.gamma = Parameter(torch.Tensor(1))
        if nonlinearity:
            self.nonlinearity = nonlinearity()
        else:
            self.nonlinearity = False
        self.reset_weights()

    def reset_weights(self):
        self.mu.data.uniform_(-0.1, 0.1)
        self.gamma.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        elements_per_set = x.size(1)
        ONE = Variable(torch.ones(elements_per_set), requires_grad=False)
        I = Variable(torch.eye(elements_per_set), requires_grad=False)
        weights = self.mu * I
        weights += self.gamma * torch.ger(ONE, ONE)
        x = torch.matmul(weights, x)

        if self.nonlinearity:
            x = self.nonlinearity(x)
        return x
