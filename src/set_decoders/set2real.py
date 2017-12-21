import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, attention_type, summary_type='mean'):
        assert attention_type in ['dot', 'cosine', 'learned']
        assert summary_type in ['mean', 'sum']
        self.attention_type = attention_type
        self.summary_type = summary_type
    
    def reset_weights(self):
        self.v.uniform_(-0.1, 0.1)

    def forward(self, x):
        batch_size, set_size, element_dim = x.size()

        # get the summary state 
        if self.summary_type == 'mean':
            summary = torch.mean(x, dim=1)
        elif self.summary_type == 'sum':
            summary = torch.sum(x, dim=1)
        
        summary = summary.repeat(1, set_size)
        summary = summary.view(batch_size, set_size, element_dim)

        # calcualte the attention
        if self.attention_type == 'dot':
            summary = summary.view(batch_size, set_size, 1, element_dim)
            x = x.view(batch_size, set_size, element_dim, 1)
            energies = torch.matmul(summary, x).view(batch_size, set_size)
        elif self.attention_type == 'cosine':
            energies = F.cosine_similarity(data, summary, dim=2)
        elif self.attention_type == 'learned':
            raise NotImplementedError('Currently not implemented')

        attention = F.softmax(energies)




class LinearSumSet(nn.Module):
    """
    Returns the linear sum of 
    every element in the set
    """
    def forward(self, x):
        return torch.sum(x, dim=1)


# class AttentionSumSet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_sum = LinearSumSet()
#         self.summary_linear = nn.Linear(4, 4)

#     def forward(self, x):
#         # calculates the summary of the data:
#         summary = self.summary_linear(self.linear_sum(x))
#         # scores = 

