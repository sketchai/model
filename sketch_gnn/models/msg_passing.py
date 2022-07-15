import torch
from typing import Dict
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.nn import MessagePassing, GINEConv, GINConv
import torch_geometric.nn as pyg_nn

import logging

from sketch_gnn.models.concatenatelinear import ConcatenateLinear
logger = logging.getLogger(__name__)


class BipartiteGINBlock(pl.LightningModule):
    """
    Bipartite Message Passing Operation with two GINConv Layers
        1) from SOURCE nodes (s) to TARGET nodes (t)
        2) from TARGET nodes (t) to SOURCE nodes (s) 
    """

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

    def forward(self, x_s, x_t, edge_index):
        x_t = self.gin_layer_s2t(x=(x_s, x_t), edge_index=edge_index)
        x_s = self.gin_layer_t2s(x=(x_t, x_s), edge_index=torch.flip(edge_index, dims=[0,]))
        return x_s, x_t


class ConcatLinearBlock(MessagePassing):

    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(emb_dim),
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.ReLU(emb_dim),
                )
        self.cat_linear = ConcatenateLinear(sizes=(emb_dim,emb_dim),output_size=emb_dim)

    def forward(self, x, edge_attr, edge_index):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.mlp(x + out)
    
    def message(self, x_j, edge_attr):
        return self.cat_linear(x_j, edge_attr).relu()


class SubnodeGINBlock(BipartiteGINBlock):
    """
    Same as BipartiteGINBLock but only one x tensor is used
    """

    def forward(self, x, edge_index):
        x = self.gin_layer_s2t(x=x, edge_index=edge_index)
        x = self.gin_layer_t2s(x=x, edge_index=torch.flip(edge_index, dims=[0,]))
        return x