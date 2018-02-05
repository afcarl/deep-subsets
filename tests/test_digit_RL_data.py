from src.datatools.numbers_data import NumbersDataset
from src.datatools.digit_RL_data import IntergersLargerThanAverage, SubsetSum
from torch.utils.data import DataLoader
import torch
import numpy as np

def test_integers_larger_than_avg():
    integers_data = IntergersLargerThanAverage(10, 5, max_integer=10)
    dataset = DataLoader(integers_data, batch_size=2)

    for i, datapoint in enumerate(dataset):
        assert tuple(datapoint.size()) == (2, 5, 8)

        int_representation = integers_data.bit_array_to_int_array(datapoint.int())
        to_be_checked = torch.stack([integers_data._get_data(2*i)[0], integers_data._get_data(2*i+1)[0]]).view(2, 5)

        assert np.allclose(to_be_checked.numpy(), int_representation)

def test_integers_larger_than_avg_reward():
    integers_data = IntergersLargerThanAverage(10, 5, max_integer=5)
    data = integers_data.int_list_to_bit_array(np.array([[5, 2, 1, 5], [3, 4, 1, 1], [5, 2, 1, 5], [5, 2, 1, 5]]))
    selected = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 0]])
    rewards = integers_data.reward_function(data, selected)
    assert tuple(rewards.size()) == (4, 1)
    assert rewards.view(-1).tolist() == [1, 2, 2, -1]

def test_integers_larger_than_avg_supervised_objective():
    integers_data = IntergersLargerThanAverage(10, 5, max_integer=10)
    bit_repr = integers_data.int_list_to_bit_array(np.array([[5, 2, 1, 5], [3, 4, 1, 1]]))
    objective = integers_data.supervised_objective(bit_repr)
    assert np.all(np.equal(objective, np.array([[1, 0, 0, 1], [1, 1, 0 ,0 ]])))

def test_subset_sum_reward():
    pass

def test_numbers_dataset_subsetting():
    numbers_dataset = NumbersDataset(10, 5, 5)
    test_set = torch.FloatTensor([[4, 3, 0, 0, 6], [4, 2, 1, 1, 0]]).view(2, 5, 1)

    subsets = numbers_dataset.subset_elements(test_set, np.array([[1, 0, 1,1, 1], [1, 0, 0, 0, 0]]), False)
    subset_1, subset_2 = subsets
    assert subset_1 == [4, 0, 0, 6]
    assert subset_2 == [4]

def test_numbers_dataset_bit_conversion():
    numbers_dataset = NumbersDataset(10, 5, 5)
    bit_repr = numbers_dataset.int_list_to_bit_array([4, 3, 1])
    assert np.allclose(bit_repr.numpy(), np.array([[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1]]))

