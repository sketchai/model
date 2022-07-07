from torch_geometric.data.lightning_datamodule import LightningDataset
import pytorch_lightning as pl
import logging
import functools
from typing import Dict

from sketch_gnn.dataloader.load import generate_dataset

from sketch_gnn.utils.logger import logger

class SketchGraphDataModule(pl.LightningDataModule):
    def __init__(self,conf: Dict = None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.prepare_data_per_node = True
        
        self.batch_size = conf.get('train').get('batch_size')
        self.d_train = conf.get('train_data')
        self.d_val = conf.get('val_data')
        self.d_test = conf.get('test_data')
    
    def train_dataloader(self):
        logger.info('-- Load Train Set')
        return generate_dataset(conf=self.d_train, batch_size=self.batch_size)

    def val_dataloader(self):
        logger.info('-- Load Validation Set')
        return generate_dataset(conf=self.d_val, batch_size=self.batch_size, sample=False)

    def test_dataloader(self):
        logger.info('-- Load Test Set')
        return generate_dataset(conf=self.d_test, batch_size=self.batch_size, sample=False)

    # def test_dataloader(self): 