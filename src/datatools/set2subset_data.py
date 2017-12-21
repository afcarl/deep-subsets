from .set2real_data import MNISTSets


class MNISTSubsets(MNISTSets):
    def __init__(self, *args, **kwargs):
        target = kwargs.get('target', None)
        assert target in ['avg', 'mean']
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data, target, labels = self._get(index)
        set_size = labels.size()
        subset = labels.float().ge(target)
        return data, subset