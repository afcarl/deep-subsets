"""
Datasets which produce sets of numbers represented in binary vectors
"""
import torch
import numpy as np
import math
from torch.utils.data import Dataset

class NumbersDataset(Dataset):
    """
    Creates dataset with numbers between 
    0 and max_integer
    """
    def __init__(self,
                 dataset_size,
                 set_size,
                 max_integer,
                 seed=0):
        """
        :param dataset_size: the size of the dataset
        :param set_size: the size of the set
        :param max_integer: the maximum value of the integers
        :param seed: the random seed to use
        """
        self.rng = np.random.RandomState(seed)
        self.dataset_size = dataset_size
        self.set_size = set_size
        self.max_integer = max_integer
        self.refresh_dataset()
    
    def refresh_dataset(self):
        # create random numbers
        data = self.rng.randint(0,
                                self.max_integer,
                                size=(self.dataset_size, self.set_size, 1),
                                dtype=np.uint8)

        # create the binary representation of it
        bit_data = np.unpackbits(data, axis=2)

        # create torch tensors:
        self.bit_data = torch.from_numpy(bit_data)
        self.data = torch.from_numpy(data)

        # check if 
        # print(self.data.size())
        # print()
        assert tuple(self.bit_data.size()) == (self.dataset_size, self.set_size, 8)
        return data

    def _get_data(self, index):
        return self.data[index], self.bit_data[index]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self._get_data(index)[1]
