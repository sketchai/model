import numpy as np
import torch
from typing import Dict
import pytorch_lightning as pl

from sketch_gnn.models.numerical_features.generator import generate_embedding
from sketch_gnn.models.dense_emb import DenseSparsePreEmbedding, ConcatenateLinear
from sketch_gnn.dataloader.batch_data import GraBatch
from sketch_gnn.utils.example_generator import ex_generator

import logging
logger = logging.getLogger(__name__)

class AttrDict(dict):
    def __init__(self, base_dict:dict):
        self.__dict__ = base_dict

class GaT(pl.LightningModule):
    """
    The neural network. Some utilitaries are included.
    """

    def __init__(self, d_model: Dict = {}, d_preprocessing_params: Dict = {}):
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

        self.init_model(d_model, d_preprocessing_params)
        # self.example_input_array = GraBatch(ex_generator())


    def init_model(self, d_model:Dict = {}, d_preprocessing_params: Dict = {}):
        embedding_dim = d_model.get('embedding_dim')
        self.lMax = d_preprocessing_params.get('lMax')
        self.embedding_dim = embedding_dim

        node_feature = generate_embedding(d_preprocessing_params.get('node_feature_dimensions'), embedding_dim)
        self.node_embedding = DenseSparsePreEmbedding(feature_embeddings= node_feature, 
                                                        fixed_embedding_cardinality=len(d_preprocessing_params.get('node_idx_map')), 
                                                        fixed_embedding_dim= embedding_dim, 
                                                        padding_idx=d_preprocessing_params.get('padding_idx'))

        edge_feature = generate_embedding(d_preprocessing_params.get('edge_feature_dimensions'), embedding_dim)
        self.edge_embedding = DenseSparsePreEmbedding(feature_embeddings=edge_feature, 
                                                        fixed_embedding_cardinality= len(d_preprocessing_params.get('edge_idx_map')), 
                                                        fixed_embedding_dim=embedding_dim)

        self.transform_edge_messages = ConcatenateLinear(embedding_dim, embedding_dim, embedding_dim)

        # Positional encoding
        if d_model.get('positional_encoding'):
            self.positional_encoding = torch.nn.Embedding(self.lMax, embedding_dim)
        else:
            self.positional_encoding = None

        # Instantiate
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=d_model.get('n_head'), dim_feedforward=2 * embedding_dim)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=d_model.get('num_layers'))

        self.prediction_edge = torch.nn.Linear(embedding_dim, 1)
        self.prediction_type = torch.nn.Linear(embedding_dim, len(d_preprocessing_params.get('edge_idx_map')) - 1)

    def forward(self, batch_data) -> Dict:
        """
            Forward function of nn
            Inputs:
                data (torch.tensor) :/
            Outputs :
                      (Dict) : a dict containing the following key : edges_pos, edges_neg and type
        """
        data = AttrDict(batch_data)

        output_transformer = self.encode_nodes(batch_data)
        edges_neg, edges_pos = data.edges_toInf_neg, data.edges_toInf_pos
        representation_final_edges = GaT.representation_final_edges(output_transformer, edges_neg, edges_pos)

        d = {"edges_pos": self.prediction_edge(representation_final_edges['edges_pos']),
             "edges_neg": self.prediction_edge(representation_final_edges['edges_neg']),
             "type": self.prediction_type(representation_final_edges['edges_pos']),
             "type_neg": self.prediction_type(representation_final_edges['edges_neg'])}
        

        return d

    def encode_nodes(self, batch_data):
        """
        Embedding + Message passing + Transformer
        """
        data = AttrDict(batch_data)

        # Compute node and edge embedding
        node_embedding = self.node_embedding(data.node_features, data.sparse_node_features)
        edge_embedding = self.edge_embedding(data.edge_features, data.sparse_edge_features)

        # Agregate node and edge information (message passing)
        agreg = self.aggregate_by_incidence(node_embedding, data.incidences, edge_embedding)
        input_embedding = node_embedding + agreg

        # Update input with positional encoding
        if self.positional_encoding is not None:
            input_embedding += self.positional_encoding(data.positions.tile(data.l_batch))

        input_embedding = input_embedding.view((data.l_batch, self.lMax, self.embedding_dim)) # reshape
        output_transformer = torch.transpose(self.transformer_encoder(torch.transpose(input_embedding, 0, 1),
                                                                      src_key_padding_mask=data.src_key_padding_mask), 0, 1)
                                                                      # Apply Transformer
        return output_transformer

    def aggregate_by_incidence(self, node_embedding, incidence, edge_embedding):

        # Select the node embedding of all the nodes that exchange messages
        edge_messages = node_embedding.index_select(0, incidence[1])

        # Create message : concatenate lineare on each node (to check)
        edge_messages = self.transform_edge_messages(edge_messages, edge_embedding)


        # Create a tensor of size node_embedding.shape[0] times list(edge_messages.shape[1:])
        output = torch.zeros([node_embedding.shape[0]] + list(edge_messages.shape[1:])).type_as(node_embedding)
        output.index_add_(0, incidence[0], edge_messages) # sum for each node on incidence[0]
        return output

    def representation_final_edges(output, edges_neg, edges_pos):
        output = output.flatten(end_dim=1)
        return {
            'edges_neg': torch.index_select(output, 0, edges_neg[:, 0]) * torch.index_select(output, 0, edges_neg[:, 1]),
            'edges_pos': torch.index_select(output, 0, edges_pos[:, 0]) * torch.index_select(output, 0, edges_pos[:, 1])}

    @torch.no_grad()
    def infer(self,batch_data)->Dict:
        data = AttrDict(batch_data)

        output = self.encode_nodes(batch_data)
        edges = data.edges
        edges_embedding = torch.index_select(output, 0, edges[:, 0]) * torch.index_select(output, 0, edges[:, 1])
        d = {
            "binary": self.prediction_edge(edges_embedding),
            "type": self.prediction_type(edges_embedding),
        }
        return d


    def loss(prediction, data, coef_neg=1., weight_types=None):
        # device = data.edges_toInf_pos_types.device
        data = AttrDict(data)
        loss_edge_pos = torch.mean(torch.nn.functional.softplus(-prediction['edges_pos']))
        loss_edge_neg = torch.mean(torch.nn.functional.softplus(prediction['edges_neg']))

        loss_type = torch.nn.functional.cross_entropy(prediction['type'], data.edges_toInf_pos_types, weight=weight_types)

        return loss_edge_pos + coef_neg * loss_edge_neg + loss_type

    def embeddings(self,batch_data)->dict:
        """returns embeddings for visualization"""
        with torch.no_grad():
            data = AttrDict(batch_data)

            embeddings = {}

            # Compute node and edge embedding
            node_embedding = self.node_embedding(data.node_features, data.sparse_node_features)
            edge_embedding = self.edge_embedding(data.edge_features, data.sparse_edge_features)

            embeddings['nodes_bf_msg_passing'] = node_embedding.cpu().detach().numpy()
            embeddings['edges_bf_msg_passing'] = edge_embedding.cpu().detach().numpy()
            # Agregate node and edge information (message passing)
            agreg = self.aggregate_by_incidence(node_embedding, data.incidences, edge_embedding)
            input_embedding = node_embedding + agreg

            embeddings['nodes_after_msg_passing'] = input_embedding.cpu().detach().numpy()
            # Update input with positional encoding
            if self.positional_encoding is not None:
                input_embedding += self.positional_encoding(data.positions.tile(data.l_batch))

            input_embedding = input_embedding.view((data.l_batch, self.lMax, self.embedding_dim)) # reshape
            output_transformer = torch.transpose(self.transformer_encoder(torch.transpose(input_embedding, 0, 1),
                                                                        src_key_padding_mask=data.src_key_padding_mask), 0, 1)  # Apply Transformer

            edges_neg, edges_pos = data.edges_toInf_neg, data.edges_toInf_pos
            representation_final_edges = GaT.representation_final_edges(output_transformer, edges_neg, edges_pos)

            embeddings['nodes_after_transformer'] = output_transformer.flatten(end_dim=1).cpu().detach().numpy()
            embeddings['edges_pos_after_transformer'] = representation_final_edges['edges_pos'].cpu().detach().numpy()
            embeddings['edges_neg_after_transformer'] = representation_final_edges['edges_neg'].cpu().detach().numpy()

            inferred_edges_pos_type = self.prediction_type(representation_final_edges['edges_pos']).cpu().detach().numpy()
            embeddings['inferred_edges_pos_type'] = np.argmax(inferred_edges_pos_type,axis=1)
        return embeddings
