import numpy as np
import torch
from typing import Dict
import pytorch_lightning as pl
import torch_geometric as pyg
from sketch_gnn.models.msg_passing import ConcatLinearBlock, SimpleConcatLinearBlock, SubnodeGINBlock, SubnodeGINResidualBlock, SubnodeLinearBlock

from sketch_gnn.models.node_embedding import NodeEmbeddingLayer
from sketch_gnn.models.concatenatelinear import ConcatenateLinear
from sketch_gnn.utils.example_generator import ex_generator

import logging
logger = logging.getLogger(__name__)

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
                    node_idx_map (dict) :
                    edge_idx_map (dict) :
                    padding_idx (int)  :
                    node_feature_dims (Dict): dictionary {primitive (int): {feature (str): dimension (int)}}
                    edge_feature_dims (Dict) : dictionary {constraint (int): {feature (str): dimension (int)}}

        embedding_dim, n_head, num_layers, do_positional_encoding : bool = True, lMax : int
        """
        # super().__init__()

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
            padding_idx=d_prep.get('edge_idx_map').get('Subnode'),
        )

        # Positional encoding
        if d_model.get('positional_encoding'):
            self.positional_encoding = torch.nn.Embedding(self.lMax, emb_dim)
        else:
            self.positional_encoding = None

        # Msg Passing
        self.n_layers = d_model.get('n_layers')
        edge_msg_passing = []
        subnode_msg_passing = []
        for _ in range(self.n_layers):
            edge_msg_passing.append(ConcatLinearBlock(emb_dim=emb_dim))
            subnode_msg_passing.append(SubnodeGINBlock(emb_dim=emb_dim))

        self.edge_msg_passing_0 = SimpleConcatLinearBlock(emb_dim=emb_dim)
        self.subnode_msg_passing_0 = SubnodeLinearBlock(emb_dim=emb_dim)

        self.edge_msg_passing = torch.nn.ModuleList(edge_msg_passing)
        self.subnode_msg_passing = torch.nn.ModuleList(subnode_msg_passing)

        self.skip_connections = d_model.get('skip_connections') or []
        if self.skip_connections:
            self.concat_linear = ConcatenateLinear(
                sizes=[emb_dim]*(len(self.skip_connections)+2),
                output_size=emb_dim)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=emb_dim,
                    nhead=4,
                    batch_first=True,
                    dim_feedforward=2 * emb_dim,
                    )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=d_model.get('transformer_enc_n_layers',2))
        

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
        x = self.node_embedding_layer(data.x)
        edge_attr_no_subnodes = data.edge_attr[data.edge_attr != data.subnode_mapping[0].item()]
        edge_attr = self.edge_embedding_layer(edge_attr_no_subnodes).repeat(2,1)

        # First Message passing
        x0 = self.edge_msg_passing_0(x, edge_attr, data.edge_index)
        x0 = self.subnode_msg_passing_0(x0, data.subnode_index)

        skip_x1 = []
        
        # Transformer
        if self.positional_encoding is not None:
            x0 = x0 + self.positional_encoding(data.positions)
        x_batched, mask = pyg.utils.to_dense_batch(x0, data.batch)
        x_batched = self.transformer(x_batched, src_key_padding_mask=~mask)
        x0 = x_batched[mask]
        
        # GIN Message Passing
        x1 = x
        for i in range(self.n_layers):
            if i in self.skip_connections:
                skip_x1.append(x1)
            x1 = self.edge_msg_passing[i](x1, edge_attr, data.edge_index)
            x1 = self.subnode_msg_passing[i](x1, data.subnode_index)

        x_cat = self.concat_linear(x0, x1, *skip_x1)
        edges_neg, edges_pos = data.constr_toInf_neg, data.constr_toInf_pos
        representation_final_edges = GaT.representation_final_edges(x_cat, edges_neg, edges_pos)
        edges_pos_repr = representation_final_edges['edges_pos']
        edges_neg_repr = representation_final_edges['edges_neg']
        return {
            "edges_pos":   self.prediction_edge(edges_pos_repr),
            "edges_neg":   self.prediction_edge(edges_neg_repr),
            "type":        self.prediction_type(edges_pos_repr),
            "type_neg":    self.prediction_type(edges_neg_repr)
            }

    def representation_final_edges(x, edges_neg, edges_pos):
        return {
            'edges_neg': torch.index_select(x, 0, edges_neg[:, 0]) * torch.index_select(x, 0, edges_neg[:, 1]),
            'edges_pos': torch.index_select(x, 0, edges_pos[:, 0]) * torch.index_select(x, 0, edges_pos[:, 1])
            }


    def loss(prediction, data, coef_neg=1., weight_types=None):
        # device = data.edges_toInf_pos_types.device
        loss_edge_pos = torch.mean(torch.nn.functional.softplus(-prediction['edges_pos']))
        loss_edge_neg = torch.mean(torch.nn.functional.softplus(prediction['edges_neg']))

        loss_type = torch.nn.functional.cross_entropy(prediction['type'], data.constr_toInf_pos_types, weight=weight_types)
        
        return loss_edge_pos, coef_neg * loss_edge_neg, loss_type

    @torch.no_grad()
    def embeddings(self, data):
        embeddings = {}
        # Compute node and edge embedding
        x = self.node_embedding_layer(data.x)
        embeddings['node_embeddings'] = x.cpu().detach().numpy()
        edge_attr_no_subnodes = data.edge_attr[data.edge_attr != data.subnode_mapping[0].item()]
        edge_attr = self.edge_embedding_layer(edge_attr_no_subnodes).repeat(2,1)

        # First Message passing
        x0 = self.edge_msg_passing_0(x, edge_attr, data.edge_index)
        x0 = self.subnode_msg_passing_0(x0, data.subnode_index)

        skip_x1 = []
        
        # Transformer
        if self.positional_encoding is not None:
            x0 = x0 + self.positional_encoding(data.positions)
        x_batched, mask = pyg.utils.to_dense_batch(x0, data.batch)
        x_batched = self.transformer(x_batched, src_key_padding_mask=~mask)
        x0 = x_batched[mask]
        
        # GIN Message Passing
        x1 = x
        for i in range(self.n_layers):
            if i in self.skip_connections:
                skip_x1.append(x1)
            x1 = self.edge_msg_passing[i](x1, edge_attr, data.edge_index)
            x1 = self.subnode_msg_passing[i](x1, data.subnode_index)

        x_cat = self.concat_linear(x0, x1, *skip_x1)
                
        embeddings['node_after_msg_passing'] = x_cat.cpu().detach().numpy()

        x_cat = self.concat_linear(x0, x1, *skip_x1)
        edges_neg, edges_pos = data.constr_toInf_neg, data.constr_toInf_pos
        representation_final_edges = GaT.representation_final_edges(x_cat, edges_neg, edges_pos)
        embeddings['edges_pos_before_classif'] = representation_final_edges['edges_pos'].cpu().detach().numpy()
        inferred_edges_pos_type = self.prediction_type(representation_final_edges['edges_pos']).cpu().detach().numpy()
        embeddings['inferred_edges_pos_type'] = np.argmax(inferred_edges_pos_type,axis=1)
        return embeddings
