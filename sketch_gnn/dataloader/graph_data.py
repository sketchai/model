from copy import copy
import numpy as np
import torch
import logging
from typing import Any

from torch_geometric.data import Data, Dataset

logger = logging.getLogger(__name__)


from sketch_gnn.utils.flat_array import load_flat_array

RNG = np.random.default_rng()


class GraphData(Data):
    """
    Graph data structure

    Main Attributes:

        x (tensor):                 (1D) node features,
        edge_attr (tensor):         (1D) edge features,
        incidences (tensor):        (n_edges,2):    ALL edges (subnodes and constraints),
        edge_index (tensor):        (2,n_constr*2):   constraints only
        subnode_index (tensor):     (2,n_subnodes*2): subnodes only,
        subnode_mapping (int)       mapping of the 'Subnode' edge

    Optional:

        positions (tensor):         (1D) positional encoding
        i_edges_given (tensor):     (1D) index of given edges
        i_edges_possible (tensor):  (1D) index of possible edges
        sequence_idx (int):         idx of sketch sequence in npy file (used for visualization)

    Attributes created for training:

        constr_toInf_pos (tensor):          (n_constr_pos,2): constraints to infer positively (label=1)
        constr_toInf_neg (tensor):          (n_constr_neg,2): constraints to infer negatively (label=0)
        constr_toInf_pos_types (tensor):    (1D): types
    """

    # The following attributes are incremented using __inc__ during batching
    _inc_x_size = [
        'constr_toInf_pos',
        'constr_toInf_neg',
        'incidences',
        'subnode_index',
        'edge_index',
    ]

    def __init__(self,
            x:torch.tensor,
            edge_attr:torch.tensor,
            incidences:torch.tensor,
            i_edges_given:torch.tensor=None,
            i_edges_possible:torch.tensor=None,
            positions:torch.tensor=None,
            edge_idx_map:dict={},
            sequence_idx:int=0,
        ):
        super().__init__()
        self.x = x
        self.edge_attr = edge_attr
        self.incidences = incidences
        self.subnode_mapping = edge_idx_map.get('Subnode')
        if x is not None:
            self._split_edge_index()
            self.num_nodes = x.shape[0]
        else:
            # This class may also be instantiated with no parameters
            self.num_nodes = 0

        self.i_edges_given = i_edges_given
        self.i_edges_possible = i_edges_possible
        self.positions = positions
        self.sequence_idx = sequence_idx

    @staticmethod
    def from_encoded_sketch(pkl_dict, edge_idx_map):
        """
        Read pickled dictionnary data
        """
        positions = torch.arange(len(pkl_dict['node_features']))
        for k, value in pkl_dict.items():
            if isinstance(value, np.ndarray):
                pkl_dict[k] = torch.from_numpy(value)
        graph = GraphData(
            x =                 pkl_dict['node_features'],
            edge_attr =         pkl_dict['edge_features'],
            incidences =        pkl_dict['incidences'],
            i_edges_given =     pkl_dict['i_edges_given'],
            i_edges_possible =  pkl_dict['i_edges_possible'],
            positions =         positions,
            edge_idx_map =      edge_idx_map,
            sequence_idx =      pkl_dict['sequence_idx'],
        )
        return graph

    def __inc__(self, key, value, *args, **kwargs):
        if key in GraphData._inc_x_size:
            return torch.tensor([self.x.size(0)])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'index' in key:
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
    
    def hide_constraints(self, prop_max_edges_given, variation):
        """
        Hide random constraints for training, should only be used once
        """
        given_constraints = self._select_constraints(self.i_edges_possible, prop_max_edges_given, variation)
        given_constraints = torch.cat([given_constraints, self.i_edges_given], dim=0)
        mask = torch.zeros(self.edge_attr.shape[0], dtype=bool)
        mask[given_constraints] = True
        
        # Create _toInf_ attributes
        self.constr_toInf_pos = self.incidences[~mask]
        self.constr_toInf_pos_types = self.edge_attr[~mask]
        adj_matrix = torch.zeros([self.x.shape[0]]*2, dtype=torch.bool)
        for node_couple in self.incidences:
            adj_matrix[node_couple[0],node_couple[1]] = True
            adj_matrix[node_couple[1],node_couple[0]] = True
        self.constr_toInf_neg = torch.nonzero(torch.triu(~adj_matrix))

        # Update existing attributes
        self.edge_attr = self.edge_attr[mask]
        self.incidences = self.incidences[mask]
        self._split_edge_index()

    def _split_edge_index(self):
        """
        Updates edge_index and subnode_index attributes
        """
        mask_without_sn = (self.edge_attr != self.subnode_mapping)
        mask_with_sn = (self.edge_attr == self.subnode_mapping)
        half_edge_index = self.incidences[mask_without_sn].T
        self.subnode_index = self.incidences[mask_with_sn].T
        self.edge_index = torch.cat([half_edge_index, half_edge_index.flip(dims=[0,])],dim=1)

    @staticmethod
    def _select_constraints(i_edges_possible, prop_max_edges_given, variation=0):
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
        return torch.tensor(curr_given_index_edges)


class GraphDataset(Dataset):
    """
    Class to store the dataset
    """
    def __init__(self,
        path_seq:str,
        path_weights:str,
        prop_max_edges_given:float,
        variation:float,
        edge_idx_map: dict,
        inference = False,
        ):
        """
        """
        super().__init__()
        self.dataset = load_flat_array(path_seq)
        if path_weights:
            self.weights = np.load(path_weights)
        self.prop_max_edges_given = prop_max_edges_given
        self.variation = variation
        self.inference = inference
        self.edge_idx_map = edge_idx_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pkl_dict = self.dataset[idx]
        g = GraphData.from_encoded_sketch(pkl_dict, self.edge_idx_map)
        if not self.inference:
            g.hide_constraints(self.prop_max_edges_given, self.variation)
        return g