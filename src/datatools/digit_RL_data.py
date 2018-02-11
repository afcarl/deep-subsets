"""
Collection of tasks for 
reinforcement learning on subset level data
`IntergersLargerThanAverage` implements a sanity check task
- the goal is to select all the elements 
  that are larger than the average

SubsetSum implements the subset sum task
- the goal is to select elements from the set
  that add upto the target

"""
import torch
import numpy as np
import math
from collections import Counter
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


class IntegersLargerThanAverage(NumbersDataset):
    def __getitem__(self, index):
        _, bit_data = self._get_data(index)
        # subset = raw_data.float().ge(raw_data.float().mean())
        return bit_data

    def reward_function(self, data, selected_elements, bit_representation=True):
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

        sets = self.subset_elements(data,
                                    selected_elements,
                                    bit_representation=bit_representation)
        if bit_representation:
            data = self.bit_array_to_int_array(data)

        rewards = []
        set_means = np.mean(data, 1)
        # sets =
        for set_mean, set_ in zip(set_means, sets):
            # +1 for correctly identifying element above mean
            # -1 for incorrect identification
            rewards.append(float((2*(np.array(set_) >= set_mean)-1).sum()))
            # -1 for everything that is wrong
            #rewards.append(float(((np.array(set_) >= set_mean)-1).sum()))
        # print(sets >= set_means)
        #
        # for set_i, data_i in zip(sets, data):
        #     # calculate what
        #     set_mean = np.mean(data_i)
        #     score = set_i >= data
        #     # set_mean = np.mean(data_i)
        #     # selected_i = np.array(data_i) >= set_mean
        #     # true_elements = self.subset_elements(np.array([data_i]).reshape(1, -1, 1), np.array([selected_i]), bit_representation=False)[0]
        #     # rewards.append(int(Counter(true_elements) == Counter(set_i)))

        return torch.FloatTensor(rewards).view(-1, 1)

    def supervised_objective(self, data):
        data = self.bit_array_to_int_array(data)
        return (data >= data.mean(1).reshape(-1, 1)).astype(int)


