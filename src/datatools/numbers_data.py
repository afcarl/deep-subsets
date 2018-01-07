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

        assert tuple(self.bit_data.size()) == (self.dataset_size, self.set_size, 8)
        return data

    def _get_data(self, index):
        return self.data[index], self.bit_data[index]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self._get_data(index)[1]

    def int_list_to_bit_array(self, list_of_numbers):
        list_of_numbers = np.array(list_of_numbers,
                                   dtype=np.uint8).reshape(1, -1, 1)
        bit_data = np.unpackbits(list_of_numbers, 2)
        return torch.from_numpy(bit_data)

    def bit_array_to_int_array(self, bit_repr):
        if type(bit_repr) != np.ndarray:
            bit_repr = bit_repr.numpy()
        
        batch_size, set_size, _ = bit_repr.shape
        return np.packbits(bit_repr, 2).reshape(batch_size, set_size)

    def subset_elements(self, data, selection_idx, bit_representation=True):
        if type(selection_idx) != np.ndarray:
            selected_elements = selection_idx.numpy()
        
        batch_size, set_size, _ = data.shape

        if bit_representation:
            numbers = self.bit_array_to_int_array(data)
        else:
            if type(data) != np.ndarray:
                numbers = data.numpy()
            else:
                numbers = data
            numbers = numbers.reshape(batch_size, set_size)

        # doing this to identify sets which are empty
        # there is a small penalty for returning empty sets but not
        # as much as returning a wrong set.
        # 1: multiply the data and indices to select elements
        #    this will only select the non-zero elements
        #    all sets will have 0 as an element
        # 2: convert the numpy array to a list of lists

        selected_elements = selected_elements.reshape(batch_size, set_size)
        sets = (numbers * selected_elements).tolist()

        # 3: filter out all the zero elements in each set
        filtered_sets = list(map(lambda set_: list(filter(lambda element: element>0, set_)), sets))

        return filtered_sets


class IntergersLargerThanAverage(NumbersDataset):
    def __getitem__(self, index):
        raw_data, bit_data = self._get_data(index)
        subset = raw_data.float().ge(raw_data.float().mean())
        return bit_data, subset



