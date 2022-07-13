import bisect
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


from sketch_gnn.utils.flat_array import load_flat_array

RNG = np.random.default_rng()

def convert_in_tensor(path:str, convert_tensor:bool):
    try:
        array = load_flat_array(path)
    except ValueError:
        array = np.load(path)
    if convert_tensor :
        return torch.tensor(array) 
    else :
        return array

def load_binary_file(path:str, convert_tensor: bool = False):
    return [convert_in_tensor(path, convert_tensor)]

        

class GraphDataset(torch.utils.data.Dataset):
    """
    Class to store the dataset; torch dataloader picks examples here. Manages the load from different files, in case of a sliced dataset.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, path_seq:str, path_weights:str, n_slice=None):
        """
        f_seqs : str, file or folder containing the preprocessed sequences. For a folder, the preprocessed slices are concatenated in a memory-efficient way;
        f_weights : str, file or folder containing the weights of the preprocessed sequences;
        n_slice: int, number of slices to concatenate, mandatory if f_seqs is a folder.
        """
        logger.debug('Load datasets...')
        self.datasets = load_binary_file(path_seq, convert_tensor=False)
        self.cumulative_sizes = self.cumsum(self.datasets)
        logger.debug('Load weights')
        if path_weights:
            self.weights = load_binary_file(path_weights, convert_tensor=True)
            if isinstance(self.weights, list):
                self.weights = torch.cat(self.weights)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


