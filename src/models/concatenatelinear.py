import torch

class ConcatenateLinear(torch.nn.Module):
    """
        A torch nn module that concatenates several inputs and mixes them using a linear layer.
        (extract from SketchGraph)
    """
    def __init__(self, left_size : int, right_size : int, output_size : int):
        super(ConcatenateLinear, self).__init__()
        self._linear = torch.nn.Linear(in_features=left_size + right_size, 
                                       out_features=output_size) # Apply a linear transform to the incoming data y = xA^T + b

    def forward(self, left, right):
        return self._linear(torch.cat((left, right), dim=-1)) # concatenation on the last dimension