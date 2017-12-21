import torch.nn as nn
from src.set_encoders import ContextBasedLinear, ContextFreeEncoder
from src.set_decoders import LinearSumSet
from src.util_layers import FlattenElements

class Set2RealNet(nn.Module):
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
            self.cbl = ContextBasedLinear(nonlinearity=nn.ReLU)
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
            x = self.cbl(x) # encode the set

        x = self.lss(x) # collapse the set

        # final rho function is a simple 2 layer NN
        x = self.lineartoh(x) 
        x = self.relu(x)
        x = self.linearout(x)
        return x
        