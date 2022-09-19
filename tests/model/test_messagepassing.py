import os
import unittest
import logging
import torch
import pickle
from sketch_gnn.models.msg_passing import SubnodeGINBlock, ConcatLinearBlock



logger = logging.getLogger(__name__)


class TestMessagePassing(unittest.TestCase):

    def setUp(self):

        self.emb_dim = 5

    def test_concatlinearblock(self):
        cl_block = ConcatLinearBlock(emb_dim=self.emb_dim)

        logger.debug(dict(cl_block.named_modules()))

        x = torch.arange(3*self.emb_dim).reshape(3,-1).float()
        edge_attr = torch.arange(2*self.emb_dim).reshape(2,-1).float()
        edge_index = torch.tensor([[0, 2],
                                  [1, 2]], dtype=int)
        logger.debug(f'x: {x}')
        logger.debug(f'edge_attr: {edge_attr}')
        logger.debug(f'edge_index: {edge_index}')
        
        # Test Message passing operation
        x_msg = cl_block.message(x_j=x[edge_index[0]], edge_attr=edge_attr)
        logger.debug(f'x_msg: {x_msg}')

        x1 = cl_block.propagate(x=x, edge_attr=edge_attr, edge_index=edge_index)
        logger.debug(f'x1: {x1}')

        x0 = torch.arange(3*self.emb_dim).reshape(3,-1).float()
        for m, i in enumerate(edge_index[1]):
            x0[i] += x_msg[m]
        torch.testing.assert_allclose(x0,x+x1)

        # Test the complete block
        x2 = cl_block(x=x, edge_attr=edge_attr, edge_index=edge_index)
        logger.debug(f'x2: {x2}')

    def test_bipartiteginblock(self):
        gin_block = SubnodeGINBlock(emb_dim=self.emb_dim)
        x = torch.arange(3*self.emb_dim).reshape(3,-1).float()
        edge_index = torch.tensor([[0, 1],
                                  [2, 2]], dtype=int)
        out = gin_block.forward(x=x, edge_index=edge_index)

        logger.debug(out)



