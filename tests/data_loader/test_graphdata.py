from collections import Counter
from copy import copy
import pickle
import unittest
import logging
import torch
import numpy as np
from sketch_gnn.utils.to_dict import parse_config
from sketch_gnn.dataloader.graph_data import GraphData, GraphDataset
from torch_geometric.data import Batch


logger = logging.getLogger()

class TestGraphData(unittest.TestCase):

    def test_creation(self):
        with open('tests/asset/dataset/mini_example/preprocessing_params.pkl','rb') as f:
            d_prep = pickle.load(f)
        edge_idx_map = d_prep['edge_idx_map']
        graph_dataset = GraphDataset(
            path_seq='tests/asset/dataset/mini_example/encoding_results.npy',
            path_weights='tests/asset/dataset/mini_example/mini_weights.npy',
            prop_max_edges_given=0.5,
            variation=0.,
            inference=True,
            edge_idx_map=edge_idx_map,
            )

        logger.debug(f'graph dataset: {graph_dataset.dataset}')
        logger.debug(f'graph dataset: {graph_dataset.weights}')

        g = graph_dataset[0]
        logger.debug('g')
        logger.debug(g)
        logger.debug('incidences')
        logger.debug(g.incidences)
        logger.debug('edge_index.T')
        logger.debug(g.edge_index.T)
        
        logger.debug('i_edges_possible')
        logger.debug(g.i_edges_possible)
        logger.debug('i_edges_given')
        logger.debug(g.i_edges_given)
        logger.debug('x')
        logger.debug(g.x)
        logger.debug('edge_attr')
        logger.debug(g.edge_attr)

        g.hide_constraints(prop_max_edges_given=0.5,variation=0.)

        logger.debug('Hiding constraints')
        logger.debug('g.edge_index')
        logger.debug(g.edge_index)
        logger.debug('g.edge_attr')
        logger.debug(g.edge_attr)

        

