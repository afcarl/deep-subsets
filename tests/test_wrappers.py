import torch
from torch.autograd import Variable
from src.datatools import RLWrapper
from src.datatools import IntegersLargerThanAverage
import numpy as np
def test_RLWrapper():
    set_sizes = [2, 3]
    datasets = [IntegersLargerThanAverage(32, set_size, 10) for set_size in set_sizes]
    environment = RLWrapper(datasets)
    data = environment.reset()
    assert isinstance(data, Variable)
    assert data.size() == (32, 2, 8) or data.size() == (32, 3, 8)
    first_batch_set_size = data.size()[1]

    random_actions = Variable(torch.IntTensor(np.random.randint(0, 2, (32, first_batch_set_size, 1))))

    next_batch, rewards, dones, info = environment.step(random_actions)
    assert isinstance(rewards, torch.Tensor)
    assert rewards.size() == (32, 1)
    assert isinstance(next_batch, Variable)
