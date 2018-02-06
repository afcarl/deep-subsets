import torch.nn as nn
from src.set_encoders import ContextBasedLinear, ContextBasedMultiChannelLinear, ContextFreeEncoder
import torch.nn.functional as F


class IntegerSubsetNet(nn.Module):
    """
    Used when the input is to be interpreted as a set of digits represented in base 2
    and the output is a probability assigned to each element of the set
    """

    def __init__(self, logprobs=True, null_model=False):
        super().__init__()

        self.null_model = null_model

        if self.null_model:
            # so many layers just to have the same number of parameters!
            cfe = nn.Sequential(nn.Linear(8, 32),
                                nn.ReLU(),
                                nn.Linear(32, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, 64),
                                nn.ReLU(),
                                nn.Linear(64,64),
                                nn.ReLU(),
                                nn.Linear(64,64),
                                nn.ReLU(),
                                nn.Linear(64,64),
                                nn.ReLU(),
                                nn.Linear(64,64),
                                nn.ReLU(),
                                nn.Linear(64,32),
                                nn.ReLU(),
                                nn.Linear(32, 16),
                                nn.ReLU(),
                                nn.Linear(16, 1))
        else:
            cfe = nn.Sequential(nn.Linear(8, 32),
                                nn.ReLU(),
                                nn.Linear(32, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
				                nn.ReLU(),
				                nn.Linear(64, 32),
				                nn.ReLU())

        self.cfe = ContextFreeEncoder(cfe, '1d')
        if not self.null_model:
            self.cbe = ContextBasedMultiChannelLinear(32, 64, nonlinearity=nn.ReLU)
            self.cbe2 = ContextBasedMultiChannelLinear(64, 64, nonlinearity=nn.ReLU)
            self.cbe3 = ContextBasedMultiChannelLinear(64, 64, nonlinearity=nn.ReLU)
            self.cbe4 = ContextBasedMultiChannelLinear(64, 64, nonlinearity=nn.ReLU)
            self.cbe5 = ContextBasedMultiChannelLinear(64, 64, nonlinearity=nn.ReLU)
            self.cbe6 = ContextBasedMultiChannelLinear(64, 32, nonlinearity=nn.ReLU)
            self.cbe7 = ContextBasedMultiChannelLinear(32, 16, nonlinearity=nn.ReLU)
            self.cbe8 = ContextBasedMultiChannelLinear(16, 1)

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
            x = self.cbe7(x)
            x = self.cbe8(x)
        if not self.logprobs:
            x = F.sigmoid(x)
        return x
