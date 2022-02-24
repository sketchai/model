import bisect
import numpy as np
import torch

from sketchgraphs_models.autoconstraint import dataset
from sketchgraphs.data import flat_array, sequence
from src.utils.maps import *

RNG = np.random.default_rng()


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

    def __init__(self, f_seqs, f_weights, n_slice=None):
        """
        f_seqs : str, file or folder containing the preprocessed sequences. For a folder, the preprocessed slices are concatenated in a memory-efficient way;
        f_weights : str, file or folder containing the weights of the preprocessed sequences;
        n_slice: int, number of slices to concatenate, mandatory if f_seqs is a folder.
        """
        if f_seqs.endswith('.npy'):
            self.datasets = [flat_array.load_flat_array(f_seqs)]
        else:
            self.datasets = []
            for i in range(n_slice):
                self.datasets.append(flat_array.load_flat_array(f_seqs + 'slice_{}_final.npy'.format(i)))
        self.cumulative_sizes = self.cumsum(self.datasets)

        if f_weights.endswith('.npy'):
            self.weights = flat_array.load_flat_array(f_weights)
        else:
            self.weights = []
            for i in range(n_slice):
                self.weights.append(torch.tensor(flat_array.load_flat_array(
                    f_seqs + 'slice_{}_weights.npy'.format(i))))
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


class GraBatch:
    """
    Class for batch instances. Manages memory moves.
    """

    def __init__(self, data):
        """
        data : as returned by the collate function.
        """
        self.l_batch = data['l_batch']
        self.node_features = data['node_features']
        self.sparse_node_features = data['sparse_node_features']
        self.incidences = data['incidences']
        self.edge_features = data['edge_features']
        self.sparse_edge_features = data['sparse_edge_features']
        self.edges_toInf_pos = data['edges_toInf_pos']
        self.edges_toInf_pos_types = data['edges_toInf_pos_types']
        self.edges_toInf_neg = data['edges_toInf_neg']
        self.src_key_padding_mask = data['src_key_padding_mask']
        self.positions = data['positions']
        self.is_given = data['is_given']

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.node_features.pin_memory()
        for k in self.sparse_node_features.keys():
            self.sparse_node_features[k]['index'].pin_memory()
            self.sparse_node_features[k]['value'].pin_memory()
        self.incidences.pin_memory()
        self.edge_features.pin_memory()
        for k in self.sparse_edge_features.keys():
            self.sparse_edge_features[k]['index'].pin_memory()
            self.sparse_edge_features[k]['value'].pin_memory()
        self.edges_toInf_pos.pin_memory()
        self.edges_toInf_pos_types.pin_memory()
        self.edges_toInf_neg.pin_memory()
        if self.src_key_padding_mask is not None:
            self.src_key_padding_mask.pin_memory()
        self.positions.pin_memory()
        return self

    # custom memory loading method on custom type
    def load_cuda_async(self, device):
        if device.type == "cpu":
            return None
        self.node_features = self.node_features.to(device=device, non_blocking=False)
        for k in self.sparse_node_features.keys():
            self.sparse_node_features[k]['index'] = self.sparse_node_features[k]['index'].to(device=device, non_blocking=False)
            self.sparse_node_features[k]['value'] = self.sparse_node_features[k]['value'].to(device=device, non_blocking=False)
        self.incidences = self.incidences.to(device=device, non_blocking=False)
        self.edge_features = self.edge_features.to(device=device, non_blocking=False)
        for k in self.sparse_edge_features.keys():
            self.sparse_edge_features[k]['index'] = self.sparse_edge_features[k]['index'].to(device=device, non_blocking=False)
            self.sparse_edge_features[k]['value'] = self.sparse_edge_features[k]['value'].to(device=device, non_blocking=False)
        self.edges_toInf_pos = self.edges_toInf_pos.to(device=device, non_blocking=False)
        self.edges_toInf_pos_types = self.edges_toInf_pos_types.to(device=device, non_blocking=False)
        self.edges_toInf_neg = self.edges_toInf_neg.to(device=device, non_blocking=False)
        if self.src_key_padding_mask is not None:
            self.src_key_padding_mask = self.src_key_padding_mask.to(device=device, non_blocking=False)
        self.positions = self.positions.to(device=device, non_blocking=False)

