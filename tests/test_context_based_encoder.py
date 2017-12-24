from src.set_encoders import ContextBasedLinear, ContextBasedMultiChannelLinear

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


cbl = ContextBasedLinear(nn.ReLU)

def test_equivariance_cbl():

    # create two data batches
    data1 = torch.ones(2, 3, 5)
    data1[0, 1].add_(1)
    data1[0, 2].add_(1)
    data1[1, 2].add_(1)

    data2 = torch.ones(*data1.size())
    data2.copy_(data1)
    # permute the first set:
    data2[0, 0] = 2
    data2[0, 1] = 1
    data2[0, 2] = 2
    

    result1 = cbl(Variable(data1))
    result2 = cbl(Variable(data2))

    # check sizes
    assert tuple(result2.size()) == tuple(result1.size())
    assert tuple(result2.size()) == tuple(data1.size())

    result1 = result1.data.numpy()
    result2 = result2.data.numpy()

    # second set doesnt change
    assert np.allclose(result1[1],result2[1])

    assert np.allclose(result1[0][0], result2[0][1])
    assert np.allclose(result1[0][1], result2[0][2])

def test_parameter_updating_cbl():
    data1 = torch.ones(2, 3, 5)
    data1[0, 1].add_(1)
    data1[0, 2].add_(1)
    data1[1, 2].add_(1)
    data1 = Variable(data1)
    target = Variable(torch.zeros(2))

    # store initial parameters
    gamma0 = cbl.gamma.data.numpy()[0]
    mu0 = cbl.mu.data.numpy()[0]


    class Loss(nn.Module):
        def forward(self, input, target):
            a = torch.sum(torch.sum(input, dim=1), dim=1)
            return torch.sum((a - target)**2)

    # prepare for training:
    criterion = Loss()
    sgd = torch.optim.SGD(cbl.parameters(), lr=0.01)

    for i in range(100):
        output = cbl(data1)
        sgd.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        sgd.step()

    mu100 = cbl.gamma.data.numpy()[0]
    gamma100 = cbl.mu.data.numpy()[0]

    # make sure that the parameters are updating:
    assert not np.allclose(mu100, mu0)
    assert not np.allclose(gamma100, gamma0)


cbml = ContextBasedMultiChannelLinear(5, 3, nn.ReLU)

def test_equivariance_cbml():

    # create two data batches
    data1 = torch.ones(2, 3, 5)
    data1[0, 1].add_(1)
    data1[0, 2].add_(1)
    data1[1, 2].add_(1)

    data2 = torch.ones(*data1.size())
    data2.copy_(data1)
    # permute the first set:
    data2[0, 0] = 2
    data2[0, 1] = 1
    data2[0, 2] = 2
    

    result1 = cbml(Variable(data1))
    result2 = cbml(Variable(data2))

    # check sizes
    assert tuple(result2.size()) == tuple(result1.size())
    data_sizes = tuple(data1.size())

    assert tuple(result2.size()) == (data_sizes[0], data_sizes[1], 3)

    result1 = result1.data.numpy()
    result2 = result2.data.numpy()

    # second set doesnt change
    assert np.allclose(result1[1],result2[1])

    assert np.allclose(result1[0][0], result2[0][1])
    assert np.allclose(result1[0][1], result2[0][2])

def test_parameter_updating_cbml():
    data1 = torch.ones(2, 3, 5)
    data1[0, 1].add_(1)
    data1[0, 2].add_(1)
    data1[1, 2].add_(1)
    data1 = Variable(data1)
    target = Variable(torch.zeros(2))

    # store initial parameters
    gamma0 = np.array(cbml.gamma.data.numpy())
    beta0 = np.array(cbml.beta.data.numpy())

    class Loss(nn.Module):
        def forward(self, input, target):
            a = torch.sum(torch.sum(input, dim=1), dim=1)
            return torch.sum((a - target)**2)

    # prepare for training:
    criterion = Loss()
    sgd = torch.optim.SGD(cbml.parameters(), lr=0.1)

    for i in range(500):
        output = cbml(data1)
        sgd.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        sgd.step()

    gamma100 = cbml.gamma.data.numpy()
    beta100 = cbml.beta.data.numpy()

    # make sure that the parameters are updating:
    assert not np.allclose(beta100, beta0)
    assert not np.allclose(gamma100, gamma0)

