import torch
from typing import Dict
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.nn import MessagePassing, GINConv
import torch_geometric.nn as pyg_nn
from .concatenatelinear import ConcatenateLinear

import logging
logger = logging.getLogger(__name__)


class BipartiteMessagePassing(MessagePassing):
    """
        Implementation of the msg passing operation
    """

    def __init__(self, embedding_dim):
        super().__init__(aggr="add")

        self.dense_merge = ConcatenateLinear(left_size=embedding_dim,right_size=embedding_dim,output_size=embedding_dim)


class GINBlock(pl.LightningModule):

    def __init__(self, emb_dim):
        super().__init__()
        self.gin_layer_s2t = self.GINLayer(emb_dim)
        self.gin_layer_t2s = self.GINLayer(emb_dim)

    @staticmethod
    def GINLayer(emb_dim, *args, **kwargs):
            ginconv = GINConv(
                nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(emb_dim),
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.ReLU(emb_dim),
                ),
                *args,
                **kwargs,
                )
            return ginconv

    def forward(self, x_p, x_c, edge_index):
        x_c = self.gin_layer_s2t(x=(x_p, x_c), edge_index=edge_index)
        x_p = self.gin_layer_t2s(x=(x_c, x_p), edge_index=torch.flip(edge_index, dims=[0,]))

        return x_p, x_c
        # self.gin_layer_s2t(, edge_index)

