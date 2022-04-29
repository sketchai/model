import torch
from typing import Dict 

from src.dataloader.graph_data import GraphDataset

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def generate_dataset(conf: Dict, batch_size:int, collate_fn:object, sample=True):

    ds = GraphDataset(path_seq=conf.get("path_data"), path_weights=conf.get("path_weights"))
    if sample:
        sampler = torch.utils.data.WeightedRandomSampler(ds.weights, len(ds.weights), replacement=True)
    else:
        sampler= None
    logger.debug(f'sampler: {ds.weights}, batch_size: {batch_size}')
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