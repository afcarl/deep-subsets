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
        cfe = nn.Sequential(nn.Linear(8, 16),
                            nn.ReLU(),
                            nn.Linear(16, 16),
                            nn.ReLU(),
                            nn.Linear(16, 8) if not null_model else nn.Linear(8, 1),
                            nn.ReLU()
                            )
        self.cfe = ContextFreeEncoder(cfe, '1d')
        if not self.null_model:
            self.cbe = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            # self.cbe2 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            # self.cbe3 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            # self.cbe4 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            self.cbe5 = ContextBasedMultiChannelLinear(8, 8, nonlinearity=nn.ReLU)
            self.cbe6 = ContextBasedMultiChannelLinear(8, 1)

        self.logprobs = logprobs

    def forward(self, x):
        x = self.cfe(x)
        if not self.null_model:
            x = self.cbe(x)
            # x = self.cbe2(x)
            # x = self.cbe3(x)
            # x = self.cbe4(x)
            x = self.cbe5(x)
            x = self.cbe6(x)
        if not self.logprobs:
            x = F.sigmoid(x)
        return x
