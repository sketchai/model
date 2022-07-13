import torch


import logging
logger = logging.getLogger(__name__)

class ConcatenateLinear(torch.nn.Module):
    """
        A torch nn module that concatenates several inputs and mixes them using a linear layer.
    """

    def __init__(self, sizes: list, output_size: int):
        """
            sizes List[int] : vector sizes
            output_size (int) : output size

            NB:  the two vectors must have the same shape (except in the concatenating dimension) or be empty
        """
        super(ConcatenateLinear, self).__init__()
        self._linear = torch.nn.Linear(in_features=sum(sizes),
                                       out_features=output_size)  # Apply a linear transform to the incoming data y = xA^T + b

    def forward(self, *tensors) -> torch.tensor:
        return self._linear(torch.cat(tensors, dim=-1))  # concatenation on the last dimension
