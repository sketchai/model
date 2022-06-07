import pytorch_lightning as pl
import logging
import functools
from typing import Dict

from sketch_gnn.dataloader.load import generate_dataset
from sketch_gnn.dataloader.collate import collate 


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

class SketchGraphDataModule(pl.LightningDataModule):
    def __init__(self,conf: Dict = None, preprocessing_params:Dict = {}):
        super().__init__()
        self.prepare_data_per_node = True
        
        self.batch_size = conf.get('train').get('batch_size')
        self.d_train = conf.get('train_data')
        self.d_val = conf.get('val_data')
        self.d_test = conf.get('test_data')
        # collate all examples in one batch
        self.collate_fn = functools.partial(collate, node_feature_dims=preprocessing_params.get('node_feature_dimensions'),
                                        edge_feature_dims=preprocessing_params.get('edge_feature_dimensions'),
                                        edge_idx_map = preprocessing_params.get('edge_idx_map'),
                                        lMax=preprocessing_params.get('lMax'),
                                        prop_max_edges_given=conf.get('train').get('prop_max_edges_given'))


    def train_dataloader(self):
        logger.info('-- Load Train Set')
        return generate_dataset(conf=self.d_train, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        logger.info('-- Load Validation Set')
        return generate_dataset(conf=self.d_val, batch_size=self.batch_size, collate_fn=self.collate_fn, sample=False)

    def test_dataloader(self):
        return generate_dataset(conf=self.d_test, batch_size=self.batch_size, collate_fn=self.collate_fn, sample=False)

    # def test_dataloader(self): 