import torch
from typing import Dict 

from sketch_gnn.dataloader.graph_data import GraphDataset

import logging
logger = logging.getLogger(__name__)

def generate_dataset(conf: Dict, batch_size:int, collate_fn:object, sample=True):

    if sample:
        ds = GraphDataset(path_seq=conf.get("path_data"), path_weights=conf.get("path_weights"))
        sampler = torch.utils.data.WeightedRandomSampler(ds.weights, len(ds.weights), replacement=True)
    else:
        ds = GraphDataset(path_seq=conf.get("path_data"), path_weights=None)
        sampler = None
    # logger.debug(f'sampler: {ds.weights}, batch_size: {batch_size}')
    # batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    # Generate a DataLoader
    return torch.utils.data.DataLoader(
                ds,
                collate_fn=collate_fn,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                # batch_sampler=batch_sampler,
                pin_memory=True,
                num_workers=conf.get('num_workers'),
                )
