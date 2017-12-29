import math
import torch
from torchvision import datasets, transforms
import numpy as np

class MNISTSets(torch.utils.data.Dataset):
    def __init__(self,
                 data_size,
                 set_sizes=list(range(4,11)),
                 target='avg',
                 train=True,
                 data_location='../data/'):
        """
        Creates a MNIST dataset where the inputs are sets of MNIST digits
        and the targets are a population level statistic of their labels.
        :param data_size: number of sets
        :param set_sizes: a list representing the number of possible elements per set 
        :param target: the population level statistic you want to predict
                        available: (avg, mean, sum, max, 2max)
        :param data_location: the location where the data is stored.
        """
        assert target in ['avg', 'mean', 'sum', 'max', '2max', 'gt5']
        datas = []
        data_per_set_size = math.ceil(data_size / len(set_sizes))
        for set_size in set_sizes:
            datas.extend(np.random.randint(0, 60000, size=(data_per_set_size, set_size)).tolist())
        
        self.data_idx = list(map(lambda idx: torch.from_numpy(np.array(idx)), datas))
        self.mnist_data = datasets.MNIST(data_location,
                                         transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Lambda(lambda x: x/255.0)]),
                                         train=train
                                        )
        self.target = target
    
    def _get(self, index):
        idx = self.data_idx[index]
        data = self.mnist_data.train_data[idx].unsqueeze(1).float()/255
        labels = self.mnist_data.train_labels[idx]
#         print(labels)
        if self.target == 'avg' or self.target == 'mean':
            target = torch.mean(labels.float())
        elif self.target == 'sum':
            target = torch.sum(labels.float())
        elif self.target == 'max':
            target = torch.max(labels.float())
        elif self.target == '2max':
            target = torch.sort(labels.float())[0][-2]
        elif self.target == 'gt5':
            target = 4.99
        return data, target, labels

    def __getitem__(self, index):
        data, target, _ = self._get(index)
        return data, target
    
    def __len__(self):
        return len(self.data_idx)

    