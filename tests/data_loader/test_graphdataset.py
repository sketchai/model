import unittest
import logging
import torch



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

import sys 
sys.path.append('/home/i37181/Documents/Projets/CAO/SketchGraphs/sketchgraphs')

from src.dataloader.graph_data import GraphDataset

class TestGraphDataset(unittest.TestCase):

    # @classmethod
    # def setUp(self):
    #     self.concatenate = ConcatenateLinear(left_size=10, right_size=1, output_size=5)

    def test_creation(self):
        graph_dataset = GraphDataset(path_seq='tests/asset/dataset/sg_test_final.npy', path_weights='tests/asset/dataset/sg_test_weights.npy', n_slice=None)

        logger.debug(f'graph dataset: {graph_dataset.datasets}')
        logger.debug(f'graph dataset: {graph_dataset.weights}')