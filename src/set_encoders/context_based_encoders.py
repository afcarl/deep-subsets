# encode elements of a set
import torch
import math
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
        stdv = 1. 
        self.mu.data.uniform_(-stdv, stdv)
        self.gamma.data.uniform_(-stdv, stdv)

    def forward(self, x):
        elements_per_set = x.size(1)
        ONE = Variable(torch.ones(elements_per_set), requires_grad=False)
        I = Variable(torch.eye(elements_per_set), requires_grad=False)
        if torch.cuda.is_available():
            ONE = ONE.cuda()
            I = I.cuda()
        weights = self.mu * I
        weights += self.gamma * torch.ger(ONE, ONE)
        x = torch.matmul(weights, x)

        if self.nonlinearity:
            x = self.nonlinearity(x)
        return x


class ContextBasedMultiChannelLinear(nn.Module):
    # implements equation 11 from appendix
    def __init__(self, input_dim, output_dim, nonlinearity=None):
        super().__init__()
        self.beta = Parameter(torch.Tensor(output_dim))
        self.gamma = Parameter(torch.Tensor(input_dim, output_dim))
        if nonlinearity:
            self.nonlinearity = nonlinearity()
        else:
            self.nonlinearity = False
        self.reset_weights()

    def reset_weights(self):
        stdv = 1. / math.sqrt(self.gamma.size(1))
        self.gamma.data.uniform_(-stdv, stdv)
        if self.beta is not None:
            self.beta.data.uniform_(-stdv, stdv)

    def forward(self, x):
        batch_size, set_size, element_dim = x.size()
        x_max = x_max = torch.max(x, dim=1)[0]
        x = x - x_max.repeat(1, 1, set_size).view(batch_size, set_size, element_dim)
        x = self.beta + torch.matmul(x, self.gamma)
        if self.nonlinearity:
            x = self.nonlinearity(x)
        return x

