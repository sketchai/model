import logging
from typing import List
import torch
from sketch_gnn.dataloader.graph_data import GraphData
from torch_geometric.data import Batch


def collate(
    batch:List[GraphData] or GraphData,
    edge_idx_map:dict,
    prop_max_edges_given=0,
    variation=0,
    *args,
    **kwargs):
    """
    Legacy function to read batch samples, batch them together and hide constraints
    """
    if isinstance(batch, GraphData):
        batch = [batch]
    l_graphs = []
    for enc_data in batch:
        graph = GraphData.from_encoded_sketch(enc_data, edge_idx_map=edge_idx_map)
        graph.hide_constraints(prop_max_edges_given,variation)
        l_graphs.append(graph)
    pyg_batch = Batch.from_data_list(
        l_graphs,
        follow_batch=['x', 'edge_attr', 'constr_toInf_pos', 'constr_toInf_neg'])
    return pyg_batch
