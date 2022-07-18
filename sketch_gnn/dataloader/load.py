import torch
from typing import Dict 

from sketch_gnn.dataloader.graph_data import GraphDataset
from torch_geometric.loader import DataLoader

import logging
logger = logging.getLogger(__name__)

def generate_dataset(conf: Dict, batch_size:int, edge_idx_map:dict):

    ds = GraphDataset(
        path_seq=conf.get("path_data"),
        path_weights=conf.get("path_weights"),
        prop_max_edges_given=conf.get("prop_max_edges_given"),
        variation=conf.get("variation"),
        edge_idx_map=edge_idx_map,
        )
    if conf.get("path_weights"):
        sampler = torch.utils.data.WeightedRandomSampler(ds.weights, len(ds.weights), replacement=True)
    else:
        sampler = None
    # logger.debug(f'sampler: {ds.weights}, batch_size: {batch_size}')
    # batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    # Generate a DataLoader
    return DataLoader(
                ds,
                follow_batch=['x', 'edge_attr', 'constr_toInf_pos', 'constr_toInf_neg'],
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                # batch_sampler=batch_sampler,
                pin_memory=True,
                num_workers=conf.get('num_workers'),
                )
