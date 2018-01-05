import torch
import numpy as np
import math
from .numbers_data import NumbersDataset

class SubsetSum(NumbersDataset):
    """
    Creates a subset sum dataset. 
    Each subset contains integer elements
    (positive or negative). 
    There may or may not be a subset whose sum is 0.
    """
    def __init__(self,
                 dataset_size,
                 set_size,
                 max_integer,
                 empty_subset_reward=-1,
                 correct_subset_reward=None,
                 target=1,
                 seed=0):
        """
        :param dataset_size: the size of the dataset
        :param set_size: the size of the set
        :param integer_range: the range of values of each integer
        :param target: the target sum needed
        :param seed: the random seed to use
        """
        super().__init__(dataset_size, set_size, max_integer, seed=seed)
        self.sum_target = target
        self.empty_subset_reward = empty_subset_reward
        self.correct_subset_reward = correct_subset_reward

    def reward_function(self, data, selected_elements=None, bit_representation=True):
        """
        Calculates the reward for picking the data
        :param data: must be a torch tensor or numpy array
                       of sizes:
                        if selected_elements is given 
                            (batch_size, set_size, 8) 
                            you can use bit_representation parameter
                            to convert automatically.
                        otherwise it should be 
                            (batch_size, set_size)
                            where you can pad the set with 0s
        :param selected_elements: the output of a neural network
                        that selects elements from 'data' 
                        based on boolean selection values of 
                        the same shape. If this is not given,
                        it is assumed that data contains the subsets
                        already
        """
        if selected_elements is not None:
            sets = self.subset_elements(data,
                                        selected_elements,
                                        bit_representation=bit_representation)
        else:
            sets = data.tolist()

        rewards = []
        for set_i in sets:
            if len(set_i) == 0: # check if empty
                rewards.append(self.empty_subset_reward)
            else:
                if not self.correct_subset_reward:
                    rewards.append(-abs(sum(set_i)-self.sum_target))
                else:
                    if sum(set_i) == self.sum_target:
                        rewards.append(self.correct_subset_reward)
                    else:
                        rewards.append(self.empty_subset_reward)
        return torch.FloatTensor(rewards)



