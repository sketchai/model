import bisect
import numpy as np
import torch
import logging
from typing import Any

from torch_geometric.data import Data

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


from sketch_gnn.utils.flat_array import load_flat_array

RNG = np.random.default_rng()


class BipartiteData(Data):
    tensors_to_offset = [
        'constr_toInf_pos',
        'constr_toInf_pos_types',
        'constr_toInf_neg',
    ]
    def __init__(self, edge_index=None, x_p=None, x_c=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.edge_index = edge_index
        self.x_p = x_p
        self.x_c = x_c
        for k, array in kwargs.items():
            self.__setattr__(key=k, value=array)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index': # shape (2,n_edges)
            return torch.tensor([[self.x_p.size(0)], [self.x_c.size(0)]])
        elif key in BipartiteData.tensors_to_offset: # shape (n_constraints,)
            return torch.tensor([self.x_c.size(0)])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'index' in key:
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
    
    @staticmethod
    def convert_to_bipartite(incidences):
        n_edges = incidences.shape[0]
        edge_index = np.empty((2,n_edges*2),dtype=np.int64)
        edge_index[0] = incidences.flatten()
        edge_index[1] = np.repeat(np.arange(n_edges), 2)
        edge_index = np.unique(edge_index, axis=1) # remove double edges for self loops
        edge_index = edge_index[:,np.argsort(edge_index[1])] # sort in same order as before
        return edge_index

    @staticmethod
    def from_encoded_sketch(pkl_dict):
        pkl_dict['edge_index'] = BipartiteData.convert_to_bipartite(pkl_dict['incidences'])
        for k, value in pkl_dict.items():
            if isinstance(value, np.ndarray):
                pkl_dict[k] = torch.from_numpy(value)
        graph = BipartiteData(
            x_p = pkl_dict['node_features'],
            x_c = pkl_dict['edge_features'].unsqueeze(1),
            incidences = pkl_dict['incidences'],
            edge_index = pkl_dict['edge_index'],
            i_edges_given = pkl_dict['i_edges_given'],
            i_edges_possible = pkl_dict['i_edges_possible'],
            sequence_idx = pkl_dict['sequence_idx'],
        )
        return graph
    
    def hide_constraints(self, prop_max_edges_given, variation):
        """
        Hide random constraints for training
        """
        given_constraints = self._select_constraints(self.i_edges_possible, self.i_edges_given, prop_max_edges_given, variation)

        mask = torch.zeros(self.x_c.shape[0], dtype=bool)
        mask[given_constraints] = True
        is_given = (self.edge_index[1] == given_constraints.unsqueeze(1)).any(axis=0)
        logger.debug(f'given {given_constraints}')
        self.edge_index = self.edge_index[:,is_given]
        self.constr_toInf_pos = self.incidences[~mask]
        self.constr_toInf_pos_types = self.x_c[~mask]

        # Compute adjacency matrix to get constr_toInf_neg
        adj_matrix = torch.zeros([self.x_p.shape[0]]*2, dtype=torch.bool)
        for node_couple in self.incidences:
            adj_matrix[node_couple[0],node_couple[1]] = True
            adj_matrix[node_couple[1],node_couple[0]] = True
        self.constr_toInf_neg = torch.nonzero(torch.triu(~adj_matrix))

    @staticmethod
    def _select_constraints(i_edges_given, i_edges_possible, prop_max_edges_given, variation=0):
        """
        Prepare a subgraph of constraints : given index edges are selected randomly among the constraint list
        """
        l = len(i_edges_possible) # compute the number of non-subnode constraints on the current ex 
        n_max_edges_given = min(int(prop_max_edges_given * l), l - 2)
        n_min_edges_given = int(n_max_edges_given*(1-variation))
        if l > 2:
            curr_given_index_edges = RNG.choice(i_edges_possible, int(RNG.uniform(n_min_edges_given, n_max_edges_given)), replace=False)
        else:
            curr_given_index_edges = np.array([], dtype=np.int64)
        curr_given_index_edges = np.concatenate([curr_given_index_edges, i_edges_given])
        return torch.tensor(curr_given_index_edges)

class GraphDataset(torch.utils.data.Dataset):
    """
    Class to store the dataset
    """
    def __init__(self,
        path_seq:str,
        path_weights:str,
        prop_max_edges_given:float,
        variation:float,        
        ):
        """
        """
        self.dataset = load_flat_array(path_seq)
        if path_weights:
            self.weights = np.load(path_weights)
        self.prop_max_edges_given = prop_max_edges_given
        self.variation = variation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pkl_dict = self.dataset[idx]
        g = BipartiteData.from_encoded_sketch(pkl_dict)
        g.hide_constraints(self.prop_max_edges_given, self.variation)
        return g

