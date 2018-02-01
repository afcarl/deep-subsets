import torch.nn as nn
from src.set_encoders import ContextBasedLinear, ContextBasedMultiChannelLinear, ContextFreeEncoder
from src.set_decoders import LinearSumSet, SimpleSubset
import torch.nn.functional as F

class IntegerSubsetNet(nn.Module):
    """
    Used when the input is to be interpreted as a set of MNIST digits
    and the output is a subset of those elements as indicated by probabilities
    """    
    def __init__(self, logprobs=True, null_model=False):
        super().__init__()

        # per element encoder is a conv neural network
        self.null_model = null_model
        cfe = nn.Sequential(nn.Linear(8, 8),
                            nn.ReLU(),
                            nn.Linear(8, 8) if not null_model else nn.Linear(8, 1))
        self.cfe = ContextFreeEncoder(cfe, '1d')
        if not self.null_model:
            self.cbe = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            self.cbe2 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            self.cbe3 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            self.cbe4 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            self.cbe5 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            self.cbe6 = ContextBasedMultiChannelLinear(8, 1)

        self.logprobs = logprobs

    def forward(self, x):
        x = self.cfe(x)
        if not self.null_model:
            x = self.cbe(x)
            x = self.cbe2(x)
            x = self.cbe3(x)
            x = self.cbe4(x)
            x = self.cbe5(x)
            x = self.cbe6(x)
        if not self.logprobs:
            x = F.sigmoid(x)
        return x
