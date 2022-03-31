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
        graph_dataset = GraphDataset(path_seq='tests/asset/dataset/mini_example/mini_final.npy', path_weights='tests/asset/dataset/mini_example/mini_weights.npy', n_slice=None)

        logger.debug(f'graph dataset: {graph_dataset.datasets}')
        logger.debug(f'graph dataset: {graph_dataset.weights}')

        for i, elt in enumerate(graph_dataset):
            if i %10 == 0 : 
                logger.debug(f'seq: {i}/{len(graph_dataset)}')

            if elt.get('length') < 10 :
                x = elt
                logger.debug(f'elt i= {i}')
                logger.debug(f'x= {x}')
                break


