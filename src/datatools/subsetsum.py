import torch
import numpy as np
import math
from torch.utils.data import Dataset

def create_instance(size, integer_range):
    return 

class SubsetSum(Dataset):
    """
    Creates a subset sum dataset. 
    Each subset contains integer elements
    (positive or negative). 
    There may or may not be a subset whose sum is 0.
    """
    def __init__(self,
                 dataset_size,
                 set_size,
                 integer_range,
                 target=0,
                 seed=0):
        """
        :param dataset_size: the size of the dataset
        :param set_size: the size of the set
        :param integer_range: the range of values of each integer
        :param target: the target sum needed
        :param seed: the random seed to use
        """
        self.rng = np.random.RandomState(seed)
        self.dataset_size = dataset_size
        self.set_size = set_size
        self.integer_range = integer_range
        self.target = target
        self.refresh_dataset()
        self.set_rewards(negative=-10, positive=0)
    
    def set_rewards(self, positive=+10, negative=-10):
        self.negative_reward = negative
        self.positive_reward = positive

    def refresh_dataset(self):
        data = self.rng.randint(-self.integer_range,
                                 self.integer_range,
                                 size=(self.dataset_size, self.set_size))

        self.data = torch.from_numpy(data)

        return data

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self.data[index]

    def solution_check(self, fullset, subset):
        is_subset = set(subset).issubset(fullset)
        subset_sum = sum(subset)
        return int(is_subset and subset_sum == self.target), is_subset, subset_sum

    def reward_function(self, fullset, subset):
        correct, is_subset, subset_sum = self.solution_check(fullset, subset)
        # print('correct?', correct)
        # print('is_subset?', is_subset)
        # print('subset_sum?',subset_sum)

        # unnecessary subset if statement? 
        # you're always picking a subset...
        if is_subset:
            # get a partial reward for returning a subset
            # based on how far you are from 0
            return -abs(self.target - subset_sum)
        else:
            # get a negative reward for not returning a subset
            return self.negative_reward

