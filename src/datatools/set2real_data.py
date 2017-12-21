import math
import torch
from torchvision import datasets, transforms
import numpy as np

class MNISTSets(torch.utils.data.Dataset):
    def __init__(self,
                 data_size,
                 set_sizes=list(range(4,11)),
                 target='avg',
                 data_location='../../data/'):
        assert target in ['avg', 'mean', 'sum', 'max', '2max']
        datas = []
        data_per_set_size = math.ceil(data_size / len(set_sizes))
        for set_size in set_sizes:
            datas.extend(np.random.randint(0, 60000, size=(data_per_set_size, set_size)).tolist())
        
        self.data_idx = list(map(lambda idx: torch.from_numpy(np.array(idx)), datas))
        self.mnist_data = datasets.MNIST(data_location,
                                         transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Lambda(lambda x: x/255.0)])
                                        )
        self.target = target
        
    def __getitem__(self, index):
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
            target = torch.sort(labels.float())[-2]
        return data, target
    
    def __len__(self):
        return len(self.data_idx)

    