import unittest
import logging
import torch
import numpy as np



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

import sys 
sys.path.append('/home/i37181/Documents/Projets/CAO/SketchGraphs/sketchgraphs')

from src.dataloader.graph_data import GraphDataset


import sys 
sys.path.append('/home/i37181/Documents/Projets/CAO/SketchGraphs/sketchgraphs')
from sketchgraphs.data import flat_array

class TestGraphDataset(unittest.TestCase):

    # @classmethod
    # def setUp(self):
    #     self.concatenate = ConcatenateLinear(left_size=10, right_size=1, output_size=5)

    def test_creation(self):
        graph_dataset = GraphDataset(path_seq='tests/asset/dataset/sg_test_final.npy', path_weights='tests/asset/dataset/sg_test_weights.npy', n_slice=None)

        logger.debug(f'graph dataset: {graph_dataset.datasets}')
        logger.debug(f'graph dataset: {graph_dataset.weights}')
        
        for ex in graph_dataset:
            logger.debug(f'element: {ex}')
            break
        path = 'tests/asset/dataset/example.npy'
        data = flat_array.save_list_flat(ex)
        np.save(path, data, allow_pickle=False)
        ex_new = np.load(path)
        res = flat_array.load_flat_array(path)
        
        logger.debug(f'element: {res[0]}')