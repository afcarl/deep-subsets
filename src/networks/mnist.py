import torch.nn as nn
from src.set_encoders import ContextBasedLinear, ContextFreeEncoder
from src.set_decoders import LinearSumSet, SimpleSubset
from src.util_layers import FlattenElements

class Set2RealNet(nn.Module):
    """
    Used when the input is to be interpreted as a Set of MNIST digits
    and the output is a real number calculated from the set
    """
    def __init__(self, encode_set=False):
        super().__init__()

        # per element encoder is a conv neural network
        cfe = nn.Sequential(nn.Conv2d(1, 1, (3, 3)),
                            nn.MaxPool2d(4),
                            nn.ReLU(),
                            nn.Conv2d(1, 1, (3, 3)),
                            nn.MaxPool2d(2),
                            nn.ReLU())
        
        self.cfe = ContextFreeEncoder(cfe, '2d')
        self.flatten = FlattenElements()
        if encode_set:
            self.cbl1 = ContextBasedLinear(nonlinearity=nn.ReLU)
            self.cbl2 = ContextBasedLinear(nonlinearity=nn.ReLU)
            self.encode_set = True
        else:
            self.encode_set = False

        self.lss = LinearSumSet()
        self.lineartoh = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.linearout = nn.Linear(4, 1)
        
    def forward(self, x):
        x = self.cfe(x) # encode individual images
        x = self.flatten(x) # flatten individual images

        if self.encode_set:
            x = self.cbl1(x) # encode the set
            x = self.cbl2(x) # encode the set

        x = self.lss(x) # collapse the set

        # final rho function is a simple 2 layer NN
        x = self.lineartoh(x) 
        x = self.relu(x)
        x = self.linearout(x)
        return x

class Seq2RealNet(nn.Module):
    """
    Used when the input is to be interpreted as a _sequence_ of MNIST digits
    and the output is a real number calculated from the sequence
    """
    def __init__(self):
        super().__init__()
        
        # per element encoder is a conv neural network
        cfe = nn.Sequential(nn.Conv2d(1, 1, (3, 3)),
                            nn.MaxPool2d(4),
                            nn.ReLU(),
                            nn.Conv2d(1, 1, (3, 3)),
                            nn.MaxPool2d(2),
                            nn.ReLU())

        self.cfe = ContextFreeEncoder(cfe, '2d')
        self.flatten = FlattenElements()

        self.lstm = nn.LSTM(4, 4, 1)
        self.relu = nn.ReLU()
        self.linearout = nn.Linear(4, 1)

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
        cfe = nn.Sequential(nn.Conv2d(1, 1, (3, 3)),
                            nn.MaxPool2d(4),
                            nn.ReLU(),
                            nn.Conv2d(1, 1, (3, 3)),
                            nn.MaxPool2d(2),
                            nn.ReLU())
        
        self.cfe = ContextFreeEncoder(cfe, '2d')
        self.flatten = FlattenElements()
        self.cbe = ContextBasedLinear(nonlinearity=nn.ReLU)
        self.cbe2 = ContextBasedLinear(nonlinearity=nn.ReLU)
        self.cbe3 = ContextBasedLinear(nonlinearity=None)
        self.final = ContextFreeEncoder(nn.Linear(4, 1), '1d')
        self.logprobs = logprobs

    def forward(self, x):
        x = self.cfe(x)
        x = self.flatten(x)
        x = self.cbe(x)
        x = self.cbe2(x)
        x = self.cbe3(x)
        x = self.final(x)
        if not self.logprobs:
            x = F.sigmoid(x)
        return x
