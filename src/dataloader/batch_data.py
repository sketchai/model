import torch 


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

