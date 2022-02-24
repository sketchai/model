import torch
from typing import Dict 

from src.dataloader.graph_data import GraphDataset

def generate_dataset(conf: Dict, batch_size:int, collate_fn:object):
    data_path: str = conf.get("path_data")
    weights_path:str = conf.get("path_weights")
    n_slice:int = conf.get("n_slice")
    num_workers: str = conf.get('num_workers')

    ds = GraphDataset(data_path, weights_path, n_slice=n_slice)

    sampler = torch.utils.data.WeightedRandomSampler(ds.weights, len(ds.weights), replacement=True)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    # Generate a DataLoader
    return torch.utils.data.DataLoader(
                ds,
                collate_fn=collate_fn,
                batch_sampler=batch_sampler,
                pin_memory=True,
                num_workers=num_workers)