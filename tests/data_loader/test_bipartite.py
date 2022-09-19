from collections import Counter
from copy import copy
import unittest
import logging
import torch
import numpy as np
logger = logging.getLogger(__name__)

from sketch_gnn.dataloader.bipartite_data import BipartiteData, GraphDataset

class TestBipartiteData(unittest.TestCase):

    def test_creation(self):
        graph_dataset = GraphDataset(
            path_seq='tests/asset/dataset/mini_example/encoding_results.npy',
            path_weights='tests/asset/dataset/mini_example/mini_weights.npy',
            prop_max_edges_given=0.5,
            variation=0.,
            inference=True,
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
        copied_g_index = copy(g.edge_index)
        selfloops = []

        # testing bipartite edge_index
        for index, couple in enumerate(g.incidences):
            if couple[0] == couple[1]:
                selfloops.append(index)
        logger.debug(selfloops)
        c = Counter(g.edge_index[1].numpy())
        for i in range(g.edge_index[1].max()):
            if i in selfloops:
                self.assertEqual(c[i], 1)
            else:
                self.assertEqual(c[i], 2, msg=i)
                
        logger.debug('i_edges_possible')
        logger.debug(g.i_edges_possible)
        logger.debug('i_edges_given')
        logger.debug(g.i_edges_given)
        logger.debug('x_p')
        logger.debug(g.x_p)
        logger.debug('x_c')
        logger.debug(g.x_c)

        g.hide_constraints(prop_max_edges_given=0.5,variation=0.)

        logger.debug('Hiding constraints')
        logger.debug('g.edge_index')
        logger.debug(g.edge_index)
        logger.debug('g.x_c')
        logger.debug(g.x_c)

        logger.debug(g.constr_toInf_neg.max())
        logger.debug(g.constr_toInf_pos.max())
        logger.debug(g.x_p.shape)


        

