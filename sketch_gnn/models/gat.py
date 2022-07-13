import numpy as np
import torch
from typing import Dict
import pytorch_lightning as pl
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing, GINConv
from sketch_gnn.models.msg_passing import GINBlock

from sketch_gnn.models.numerical_features.generator import generate_embedding
from sketch_gnn.models.node_embedding import NodeEmbeddingLayer
from sketch_gnn.models.concatenatelinear import ConcatenateLinear
from sketch_gnn.utils.example_generator import ex_generator

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

class GaT(pl.LightningModule):
    """
    The neural network. Some utilitaries are included.
    """

    def __init__(self, d_model: Dict = {}, d_prep: Dict = {}):
        """

        embedding_dim (int) :  the embedding dimension used in the whole network;
        n_head (int)     : number of heads in the multi-head attention mecanism;
        num_layers (int) : number of layers for the encoder;
        do_positional_encoding: bool, set to True for adding a positional encoding;
        d_preprocessing_params (Dict): a dict returned by the preprocessing in 'preprocessing_params.pkl'.
                It must contains the following keys :
                    lMax: int, length of the examples
                    node_idx_map (int) :
                    edge_idx_map (int) :
                    padding_idx (int)  :
                    node_feature_dims (Dict): dictionary {primitive (int): {feature (str): dimension (int)}}
                    edge_feature_dims (Dict) : dictionary {constraint (int): {feature (str): dimension (int)}}

        embedding_dim, n_head, num_layers, do_positional_encoding : bool = True, lMax : int
        """
        super().__init__()

        self.lMax = d_prep.get('lMax')
        emb_dim = d_model.get('embedding_dim')
        self.embedding_dim = emb_dim

        self.node_embedding_layer = NodeEmbeddingLayer(
            embedding_dim=emb_dim,
            feature_dims=d_prep.get('node_feature_dimensions'),
            node_idx_map=d_prep.get('node_idx_map'),
        )

        self.edge_embedding_layer = torch.nn.Embedding(
            embedding_dim=emb_dim,
            num_embeddings=len(d_prep.get('edge_idx_map')),
        )

        # Positional encoding
        if d_model.get('positional_encoding'):
            self.positional_encoding = torch.nn.Embedding(self.lMax, emb_dim)
        else:
            self.positional_encoding = None

        # Msg Passing
        gin_blocks = []
        for _ in range(d_model.get('n_layers')):
            gin_blocks.append(GINBlock(emb_dim=emb_dim))
        self.gin_blocks = torch.nn.ModuleList(gin_blocks)

        self.skip_connections = d_model.get('skip_connections')
        self.prediction_edge = torch.nn.Linear(emb_dim, 1)
        self.prediction_type = torch.nn.Linear(emb_dim, len(d_prep.get('edge_idx_map')) - 1)

    def forward(self, data) -> Dict:
        """
            Forward function of nn
            Inputs:
                data  :/
            Outputs :
                      (Dict) : a dict containing the following key : edges_pos, edges_neg and type
        """

        # Compute node and edge embedding
        x_p = self.node_embedding_layer(data.x_p)
        x_c = self.edge_embedding_layer(data.x_c)

        # Update input with positional encoding
        if self.positional_encoding is not None:
            x_p += self.positional_encoding(data.positions)
        

        for step, gin_step in enumerate(self.gin_blocks):
            x_p, x_c = gin_step(x_p, x_c, data.edge_index)
            # if step in self.skip_connections:                
            #     pass

        # Message Passing
        edges_neg, edges_pos = data.constr_toInf_neg, data.constr_toInf_pos

        representation_final_edges = GaT.representation_final_edges(x_p, edges_neg, edges_pos)

        edges_pos_repr = representation_final_edges['edges_pos']
        edges_neg_repr = representation_final_edges['edges_neg']

        d = {"edges_pos":   self.prediction_edge(edges_pos_repr),
             "edges_neg":   self.prediction_edge(edges_neg_repr),
             "type":        self.prediction_type(edges_pos_repr),
             "type_neg":    self.prediction_type(edges_neg_repr)}
        return d

    def representation_final_edges(x_p, edges_neg, edges_pos):
        return {
            'edges_neg': torch.index_select(x_p, 0, edges_neg[:, 0]) * torch.index_select(x_p, 0, edges_neg[:, 1]),
            'edges_pos': torch.index_select(x_p, 0, edges_pos[:, 0]) * torch.index_select(x_p, 0, edges_pos[:, 1])}


    def loss(prediction, data, coef_neg=1., weight_types=None):
        # device = data.edges_toInf_pos_types.device
        loss_edge_pos = torch.mean(torch.nn.functional.softplus(-prediction['edges_pos']))
        loss_edge_neg = torch.mean(torch.nn.functional.softplus(prediction['edges_neg']))

        loss_type = torch.nn.functional.cross_entropy(prediction['type'], data.constr_toInf_pos_types, weight=weight_types)

        return loss_edge_pos + coef_neg * loss_edge_neg + loss_type

