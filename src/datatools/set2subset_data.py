from .set2real_data import MNISTSets
from .numbers_data import NumbersDataset

class MNISTSubsets(MNISTSets):
    def __init__(self, *args, **kwargs):
        target = kwargs.get('target', 'avg')
        assert target in ['avg', 'mean', 'gt5']

        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data, target, labels = self._get(index)
        set_size = labels.size()
        subset = labels.float().ge(target)
        return data, subset

class IntegerSubsetsSupervised(NumbersDataset):
    def __init__(self, dataset_size, set_size, max_integer, target, **kwargs):
        assert target in ['mean', 'gt5']
        self.target = target
        super().__init__(dataset_size, set_size, max_integer, **kwargs)

    def __getitem__(self, index):
        """
        This returns a set of binary represented digits
        and the target to be matched.
        :param index:
        :return:
        """

        base_10, base_2 = self._get_data(index)

        if self.target == 'mean':
            y_target = base_10 >= base_10.mean()
        elif self.target == 'gt5':
            y_target = base_10 >= 5.0
        else:
            raise ValueError('Unknown Target {}'.format(self.target))

        return base_2, y_target


class IntegerSubsetSumSupervised(NumbersDataset):
    def __init__(self):
        pass