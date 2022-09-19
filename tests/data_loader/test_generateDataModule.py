import unittest
import logging
import os
import pickle
from sketch_gnn.utils.to_dict import parse_config
logger = logging.getLogger()

from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule

class TestSketchGraphDataModule(unittest.TestCase):

    def test_creation(self):

        conf = parse_config('tests/asset/mock/gat_example.yml')
        graph_dataset = SketchGraphDataModule(conf)

        logger.debug(f'graph dataset: {graph_dataset}')

        train_dataset = graph_dataset.train_dataloader()
        logger.debug(f'train_dataset: nb size={len(train_dataset)}')
        logger.debug(f'train_dataset.dataset: nb size={len(train_dataset.dataset)}')
        # logger.debug(f'train_dataset: {train_dataset}')
        self.assertTrue(len(train_dataset) > 0)

        val_dataset = graph_dataset.val_dataloader()
        logger.debug(f'val_dataset: nb size={len(val_dataset)}')
        logger.debug(f'val_dataset.dataset: nb size={len(val_dataset.dataset)}')
        # logger.debug(f'val_dataset: {val_dataset}')
        self.assertTrue(len(val_dataset) > 0)
        
        logger.debug(len(val_dataset))
        for i, batch in enumerate(val_dataset):
            if i > 2 :
                break
            logger.debug(f'batch = {batch} {i} /{len(val_dataset)}')
            logger.debug(f'batch.batch = {batch.batch}')
            logger.debug(f'batch.edge_index = {batch.edge_index}')
            # logger.debug(f'{i} batch:{batch}')
