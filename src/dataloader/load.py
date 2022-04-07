import torch
from typing import Dict 

from src.dataloader.graph_data import GraphDataset

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def generate_dataset(conf: Dict, batch_size:int, collate_fn:object):

    ds = GraphDataset(path_seq=conf.get("path_data"), path_weights=conf.get("path_weights"), n_slice=conf.get("n_slice"))

    sampler = torch.utils.data.WeightedRandomSampler(ds.weights, len(ds.weights), replacement=True)
    logger.debug(f'sampler: {ds.weights}, batch_size: {batch_size}')
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    # Generate a DataLoader
    return torch.utils.data.DataLoader(
                ds,
                collate_fn=collate_fn,
                batch_sampler=batch_sampler,
                pin_memory=True,
                num_workers=conf.get('num_workers'))