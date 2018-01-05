import torch.nn as nn
from src.set_encoders import ContextBasedLinear, ContextBasedMultiChannelLinear, ContextFreeEncoder
from src.set_decoders import LinearSumSet, SimpleSubset
import torch.nn.functional as F

class IntegerSubsetNet(nn.Module):
    """
    Used when the input is to be interpreted as a set of MNIST digits
    and the output is a subset of those elements as indicated by probabilities
    """    
    def __init__(self, logprobs=True):
        super().__init__()

        # per element encoder is a conv neural network
        
        cfe = nn.Linear(8, 8)
        self.cfe = ContextFreeEncoder(cfe, '1d')
        self.cbe = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
        self.cbe2 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
        self.cbe3 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
        self.cbe4 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
        self.cbe5 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
        self.cbe6 = nn.LSTM(8, 4, 1, bidirectional=True)
        self.decoder = nn.LSTM(8, 1, 1)
        self.logprobs = logprobs

    def forward(self, x):
        x = self.cfe(x)
        x = self.cbe(x)
        x = self.cbe2(x)
        x = self.cbe3(x)
        x = self.cbe4(x)
        x = self.cbe5(x)
        x = x.permute(1, 0, 2)
        x = self.cbe6(x)[0]
        x = self.decoder(x)
        x = x[0].permute(1, 0, 2)
        if not self.logprobs:
            x = F.sigmoid(x)
        return x
