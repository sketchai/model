import pytorch_lightning as pl
import logging
import functools
from typing import Dict

from sketch_gnn.dataloader.load import generate_dataset
from sketch_gnn.dataloader.collate import collate 



logger = logging.getLogger(__name__)

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
                                        lMax=preprocessing_params.get('lMax'))



    def train_dataloader(self):
        logger.info('-- Load Train Set')
        collate_fn = functools.partial(
            self.collate_fn,
            prop_max_edges_given=self.d_train.get('prop_max_edges_given'),
            variation=self.d_train.get('variation'))
        return generate_dataset(conf=self.d_train, batch_size=self.batch_size, collate_fn=collate_fn)

    def val_dataloader(self):
        logger.info('-- Load Validation Set')
        collate_fn = functools.partial(
            self.collate_fn,
            prop_max_edges_given=self.d_val.get('prop_max_edges_given'),
            variation=self.d_val.get('variation'))
        return generate_dataset(conf=self.d_val, batch_size=self.batch_size, collate_fn=collate_fn, sample=False)

    def test_dataloader(self):
        collate_fn = functools.partial(
            self.collate_fn,
            prop_max_edges_given=self.d_test.get('prop_max_edges_given'),
            variation=self.d_test.get('variation'))
        return generate_dataset(conf=self.d_test, batch_size=self.batch_size, collate_fn=collate_fn, sample=False)

    # def test_dataloader(self): 