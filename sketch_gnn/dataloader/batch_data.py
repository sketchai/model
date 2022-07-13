import torch 
import logging 

logger = logging.getLogger(__name__)

class GraBatch:
    """
    Class for batch instances. Manages memory moves.
    """

    def __init__(self, data):
        """
        data : as returned by the collate function.
        """
        # logger.info(f'data: {data}')

        

        self.l_batch = data.get('l_batch')
        self.node_features = data.get('node_features')
        self.sparse_node_features = data.get('sparse_node_features')
        self.incidences = data.get('incidences')
        self.edge_features = data.get('edge_features')
        self.sparse_edge_features = data.get('sparse_edge_features')
        self.edges_toInf_pos = data.get('edges_toInf_pos')
        self.edges_toInf_pos_types = data.get('edges_toInf_pos_types')
        self.edges_toInf_neg = data.get('edges_toInf_neg')
        self.src_key_padding_mask = data.get('src_key_padding_mask')
        self.positions = data.get('positions')
        self.is_given = data.get('given_index_edges')

        

    def __repr__(self):
        rep = f'l_batch: {self.l_batch} \n'
        rep = rep + f'node_features: {self.node_features} \n'
        rep = rep + f'sparse_node_features: {self.sparse_node_features} \n'
        rep = rep + f'incidences: {self.incidences} \n'
        rep = rep + f'edge_features: {self.edge_features} \n'
        rep = rep + f'sparse_edge_features: {self.sparse_edge_features} \n'
        rep = rep + f'edges_toInf_pos: {self.edges_toInf_pos} \n'
        rep = rep + f'edges_toInf_neg: {self.edges_toInf_neg} \n'
        rep = rep + f'src_key_padding_mask: {self.src_key_padding_mask} \n'
        rep = rep + f'positions: {self.positions} \n'
        rep = rep + f'is_given: {self.is_given} \n'
        return rep



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

