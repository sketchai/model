import unittest
import logging
import torch
import pickle
from src.utils.to_dict import yaml_to_dict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

import sys 
sys.path.append('/home/i37181/Documents/Projets/CAO/SketchGraphs/sketchgraphs')

from src.dataloader.generate_dataModule import SketchGraphDataModule

class TestGraphDataset(unittest.TestCase):

    # @classmethod
    # def setUp(self):
    #     self.concatenate = ConcatenateLinear(left_size=10, right_size=1, output_size=5)

    def test_creation(self):

        conf = yaml_to_dict('config/gat.yml')
        d_train = conf.get('train')
        with open(d_train.get('prep_parms_path'), 'rb') as f:
            d_prep = pickle.load(f)
        graph_dataset = SketchGraphDataModule(conf, d_prep)

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
        
        for i, batch in enumerate(val_dataset):
            logger.debug(f'{i} /{len(val_dataset)}')
            # logger.debug(f'{i} batch:{batch}')

# piste pour le collate function: https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html