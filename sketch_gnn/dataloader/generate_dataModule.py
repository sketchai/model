import pickle
from torch_geometric.data.lightning_datamodule import LightningDataModule
import pytorch_lightning as pl
import logging
import functools
from typing import Dict

from sketch_gnn.dataloader.load import generate_dataset

logger = logging.getLogger(__name__)

class SketchGraphDataModule(LightningDataModule):

    def __init__(self,conf: Dict ,*args, **kwargs):
        super().__init__(has_test=True, has_val=True, *args,**kwargs)
        self.prepare_data_per_node = True
        
        self.batch_size = conf.get('train').get('batch_size')
        self.d_train = conf.get('train_data')
        self.d_val = conf.get('val_data')
        self.d_test = conf.get('test_data')
        with open(conf.get('prep_parms_path'),'rb') as f:
            d_prep = pickle.load(f)
        self.edge_idx_map = d_prep['edge_idx_map']

    def train_dataloader(self):
        logger.info('-- Load Train Set')
        return generate_dataset(conf=self.d_train, batch_size=self.batch_size, edge_idx_map=self.edge_idx_map)

    def val_dataloader(self):
        logger.info('-- Load Validation Set')
        return generate_dataset(conf=self.d_val, batch_size=self.batch_size, edge_idx_map=self.edge_idx_map)

    def test_dataloader(self):
        logger.info('-- Load Test Set')
        return generate_dataset(conf=self.d_test, batch_size=self.batch_size, edge_idx_map=self.edge_idx_map)

    def _kwargs_repr(self, *args, **kwargs):
        return self.__getattribute__('kwargs')