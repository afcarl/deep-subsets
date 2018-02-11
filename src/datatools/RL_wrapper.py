from torch.autograd import Variable
from torch.utils.data import DataLoader
import random

class Wrapper(object):
    pass

class RLWrapper(Wrapper):
    _pytorch = True
    _vectorized = True
    def __init__(self, datasets, batch_size=32, workers=1, use_cuda=False):
        """
        :param datasets: A list of DataSet objects
        :param batch_size: the batch size (i.e. number of agents interacting)
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.workers = workers
        self.use_cuda = use_cuda
        self.most_recent_batch = None
        # self.reset()

    def reset_task(self):
        [dataset.refresh_dataset() for dataset in self.datasets]
        pass
    def _variable_wrap(self, tensor):
        variable = Variable(tensor)
        if self.use_cuda:
            variable = variable.cuda()
        return variable

    def get_data(self):
        try:
            batch = next(self.iterator)
            self.most_recent_batch = batch
            return self._variable_wrap(batch)
        except StopIteration:
            self.most_recent_batch = None
            return False


    def reset(self):
        self.current_dataset = random.sample(self.datasets, 1)[0]
        self.iterator = iter(DataLoader(self.current_dataset, batch_size=self.batch_size, num_workers=self.workers))
        return self.get_data()

    def step(self, action):
        assert self.most_recent_batch is not None
        # reward is the proportion of elements correct?
        if isinstance(action, Variable):
            action = action.data
        action = action.cpu().int()
        rewards = self.current_dataset.reward_function(self.most_recent_batch.int(), action)
        next_batch = self.get_data()
        if type(next_batch) is bool and next_batch == False:
            next_batch = self.reset() #TODO should the user reset manually or should it be handled internally?
        return next_batch, rewards, [True]*self.most_recent_batch.size()[0], {'set_size': next_batch.size()[1]}