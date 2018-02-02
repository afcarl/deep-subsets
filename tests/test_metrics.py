from src.metrics import set_accuracy
from src.datatools.set2subset_data import IntegerSubsetsSupervised
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable

def test_set_accuracy():
    dset = IntegerSubsetsSupervised(10, 5, 10, 'mean')
    dl = DataLoader(dset, batch_size=2)
    _, (x, y) = next(enumerate(dl))
    y_hat = torch.FloatTensor([[0, 1, 0, 1, 1], [1, 0, 0, 1, 1]])
    set_acc, elem_acc = set_accuracy(y, y_hat)

    assert set_acc == 0.5
    assert elem_acc == 0.9

    y_hat = torch.FloatTensor([[0, 1, 0, 1, 1], [1, 0, 0, 1, 0]])
    set_acc, elem_acc = set_accuracy(y, y_hat)

    assert set_acc == 1, 'Set acc was {}'.format(set_acc)
    assert elem_acc == 1


