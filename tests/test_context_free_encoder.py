from src.set_encoders import ContextFreeEncoder

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Simple2DConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def test_1d_inputs():
    # 2 sets
    # 3 elements per set
    # (1, 10) features per element per set
    x = torch.ones(2, 3, 1, 10)
    x[:, 1] = x[:, 1] + 1
    x[1, 2] = x[1, 2] + 1

    cfe = ContextFreeEncoder(nn.Conv1d(1, 2, 2), '1d')
    y = cfe(Variable(x))
    assert tuple(y.size()) == (2, 3, 2, 9)
    y = y.view(2, 3, 2*9).data.numpy()
    assert np.allclose(y[0, 1], y[1, 1])
    assert np.allclose(y[0, 0], y[0, 2])
    assert np.allclose(y[0, 1], y[1, 2])

def test_2d_inputs():
    # 2 sets
    # 3 images per set
    # 3 channels per image
    # 32x32 is image size
    pix = torch.ones(2, 3, 3, 32, 32)
    pix[0][0].add_(1)
    pix[0][2].add_(1)
    pix[1][2].add_(1)

    cfe = ContextFreeEncoder(Simple2DConvNet(), '2d')
    cfe.train(False) # stop stochastic dropouts
    y = cfe(Variable(pix))
    # since this is a flattened extractor we should get:
    assert tuple(y.size()) == (2, 3, 10)
    y = y.data.numpy()
    assert np.allclose(y[0][0], y[0][2])
    assert np.allclose(y[0][1], y[1][1])
    assert np.allclose(y[0][0], y[1][2])
