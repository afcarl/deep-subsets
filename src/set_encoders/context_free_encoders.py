import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextFreeEncoder(nn.Module):
    def __init__(self, element_encoder, element_dims='1d'):
        """
        Applies a module on each element of the set independently.

        WARNING: only tested with convolution-like encoders
        :param element_encoder: the encoder to apply to each element of the set
        :param element_dims: the type of element in the set (can be 1d or 2d)
        :returns Variable: flattened extracted features
        Example:
        If your set contains 2D images and you 
        want to use a Conv2D to extract features
        ```
        element_encoder = nn.Conv2D(3, 1, (2,2))
        set_encoder = ContextFreeEncoder(element_encoder, '2d')
        
        # inputs:
        # batch size x number of elements in set x channels x height x width
        
    
        # outputs:
        # batch_size x number of elements in set x -1
        ```
        """
        super().__init__()
        assert element_dims in ['1d', '2d']
        self.element_encoder = element_encoder
        self.element_dims = element_dims

    def reshape_input(self, input):
        """
        Reshapes the inputs so that each element of the set
        is represented as an element in the batch
        """
        if self.element_dims == '1d':
            sizes = input.size()
            if len(sizes) == 4:
                batch_size, set_size, channels, L = sizes
                reshaped = input.view(batch_size * set_size, channels, L)
            elif len(sizes) == 3:
                batch_size, set_size, L = sizes
                reshaped = input.view(batch_size * set_size, L)
            return reshaped, (batch_size, set_size)
        elif self.element_dims == '2d':
            batch_size, set_size, channels, H, W = input.size()
            reshaped = input.view(batch_size * set_size, channels, H, W)
            return reshaped, (batch_size, set_size)
        else:
            raise ValueError('Unknown input size.')

    def reshape_output(self, output, batch_size, set_size):
        """
        Reshapes the output so that the elements from the batch
        are placed as elements of their respective sets.
        """

        output_sizes = output.size()
        # print('output_sizes:',output_sizes)
        reshaped = output.view(batch_size, set_size, *output_sizes[1:])
        return reshaped

    def forward(self, x):
        x, sizes = self.reshape_input(x)
        x = self.element_encoder(x)
        # print('sizes:',sizes)
        return self.reshape_output(x, *sizes)


