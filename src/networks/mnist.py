import torch.nn as nn
from src.set_encoders import (
    ContextBasedLinear,
    ContextBasedMultiChannelLinear,
    ContextFreeEncoder)
from src.set_decoders import LinearSumSet, SimpleSubset
from src.util_layers import FlattenElements
import torch.nn.functional as F

class ElementFlatten(nn.Module):
    def forward(self, x):
        batch_size, dim1, dim2, dim3 = x.size()
        return x.view(batch_size, dim1*dim2*dim3)


def get_MNIST_extractor():
    # from the keras MNIST example: 
    # https://github.com/keras-team/keras/blob/12a060f63462f2e5f838b70cadb2079b4302f449/examples/mnist_cnn.py#L47-L57
    return nn.Sequential(nn.Conv2d(1, 32, (3, 3)),
                         nn.ReLU(),
                         nn.Conv2d(32, 32, (3, 3)),
                         nn.ReLU(),
                         nn.MaxPool2d((2,2)),
                         ElementFlatten(),
                         nn.Linear(4608, 128),
                         nn.ReLU())

class Set2RealNet(nn.Module):
    """
    Used when the input is to be interpreted as a Set of MNIST digits
    and the output is a real number calculated from the set
    """
    def __init__(self, encode_set=False):
        super().__init__()

        # per element encoder is a conv neural network
        cfe = get_MNIST_extractor()
        
        self.cfe = ContextFreeEncoder(cfe, '2d')
        self.flatten = FlattenElements()
        # if encode_set:
        #     self.cbl1 = ContextBasedLinear(nonlinearity=nn.ReLU)
        #     self.cbl2 = ContextBasedLinear(nonlinearity=nn.ReLU)
        #     self.encode_set = True
        # else:
        self.encode_set = False

        self.lss = LinearSumSet()
        # self.lineartoh1 = nn.Linear(128, 128)
        # self.relu = nn.ReLU()
        # self.lineartoh2 = nn.Linear(128, 32)
        # self.relu = nn.ReLU()
        # self.linearout = nn.Linear(32, 1)


        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 100)
        self.l3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.cfe(x) # encode individual images
        x = self.flatten(x) # flatten individual images
        x = self.lss(x) # collapse the set

        # final rho function is a simple 2 layer NN
        x = self.l1(x) 
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

class Seq2RealNet(nn.Module):
    """
    Used when the input is to be interpreted as a _sequence_ of MNIST digits
    and the output is a real number calculated from the sequence
    """
    def __init__(self):
        super().__init__()
        
        # per element encoder is a conv neural network
        cfe = get_MNIST_extractor()

        self.cfe = ContextFreeEncoder(cfe, '2d')
        self.flatten = FlattenElements()

        # self.lstm = nn.LSTM(128, 64, 1)
        # self.relu = nn.ReLU()
        # self.linearout = nn.Linear(64, 1)

        # attempt to make it more equal to the set based method
        self.lstm = nn.LSTM(128, 32, 2)
        self.relu = nn.ReLU()
        self.linearout = nn.Linear(32, 1)

    def forward(self, x):
        x = self.cfe(x)
        x = self.flatten(x)
        x = x.permute(1, 0, 2)
        hT, _ = self.lstm(x)
        x = hT[-1]
        x = self.relu(x)
        x = self.linearout(x)

        return x
        
class Set2SubsetNet(nn.Module):
    """
    Used when the input is to be interpreted as a set of MNIST digits
    and the output is a subset of those elements as indicated by probabilities
    """    
    def __init__(self, logprobs=True):
        super().__init__()

        # per element encoder is a conv neural network
        cfe = get_MNIST_extractor()
        
        self.cfe = ContextFreeEncoder(cfe, '2d')
        self.cbe = ContextBasedMultiChannelLinear(128, 128, nonlinearity=nn.ReLU)
        self.cbe2 = ContextBasedMultiChannelLinear(128, 64, nonlinearity=nn.ReLU)
        self.cbe3 = ContextBasedMultiChannelLinear(64, 1, nonlinearity=None)
        self.logprobs = logprobs

    def forward(self, x):
        x = self.cfe(x)
        x = self.cbe(x)
        x = self.cbe2(x)
        x = self.cbe3(x)
        if not self.logprobs:
            x = F.sigmoid(x)
        return x

class Set2SubsetNetNull(nn.Module):
    def __init__(self, logprobs=True):
        super().__init__()
        mnist_extractor = get_MNIST_extractor() 
        classifier = nn.Sequential(mnist_extractor,
                          nn.Linear(128, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 1))
        self.cfe = ContextFreeEncoder(classifier, '2d')
        self.logprobs = logprobs

    def forward(self, x):
        x = self.cfe(x)
        if not self.logprobs:
            x = F.sigmoid(x)
        return x

# def Seq2SubsetNet(nn.Module):
#     """
#     Used when the input is to be interpreted as a sequence of MNIST digits
#     and the output is a subset of those elements as indicated by probabilities
#     """

#     def __init__(self, logprobs=True):
#         super().__init__()
#         # per element encoder is a conv neural network
#         cfe = get_MNIST_extractor()
        
#         self.cfe = ContextFreeEncoder(cfe, '2d')
#         self.flatten = FlattenElements()
#         self.lstm = nn.LSTM(...)
        