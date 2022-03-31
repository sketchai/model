import numpy as np
import torch
from typing import Dict
import pytorch_lightning as pl

from src.models.numerical_features.generator import generate_embedding
from src.models.dense_emb import DenseSparsePreEmbedding, ConcatenateLinear

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

from src.utils.maps import NODE_IDX_MAP, EDGE_IDX_MAP, PADDING_IDX

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
        self.prediction_type = torch.nn.Linear(embedding_dim, len(d_preprocessing_params.get('edge_idx_map')))

    def forward(self, data) -> Dict:
        """
            Forward function of nn
            Inputs:
                data (torch.tensor) :/
            Outputs :
                      (Dict) : a dict containing the following key : edges_pos, edges_neg and type
        """

        # Compute node and edge embedding
        node_embedding = self.node_embedding(data.node_features, data.sparse_node_features)
        edge_embedding = self.edge_embedding(data.edge_features, data.sparse_edge_features)

        # Agregate node and edge information (message passing)
        agreg = self.aggregate_by_incidence(node_embedding, data.incidences, edge_embedding)
        input_embedding = node_embedding + agreg

        # Update input with positional encoding
        if self.positional_encoding is not None:
            input_embedding += self.positional_encoding(data.positions.tile(data.l_batch).cuda())

        input_embedding = input_embedding.view((data.l_batch, self.lMax, self.embedding_dim))
        output_transformer = torch.transpose(self.transformer_encoder(torch.transpose(input_embedding, 0, 1),
                                                                      src_key_padding_mask=data.src_key_padding_mask.cuda()), 0, 1)  # Apply Transformer

        edges_neg, edges_pos = data.edges_toInf_neg.cuda(), data.edges_toInf_pos.cuda()
        representation_final_edges = GaT.representation_final_edges(output_transformer, edges_neg, edges_pos)

        d = {"edges_pos": self.prediction_edge(representation_final_edges['edges_pos']),
             "edges_neg": self.prediction_edge(representation_final_edges['edges_neg']),
             "type": self.prediction_type(representation_final_edges['edges_pos'])}
        

        return d

    def aggregate_by_incidence(self, node_embedding, incidence, edge_embedding):

        edge_messages = node_embedding.index_select(0, incidence[1].cuda())
        edge_messages = self.transform_edge_messages(edge_messages, edge_embedding)

        output = node_embedding.new_zeros([node_embedding.shape[0]] + list(edge_messages.shape[1:]))
        output.index_add_(0, incidence[0].cuda(), edge_messages.cuda())
        return output

    def representation_final_edges(output, edges_neg, edges_pos):
        output = output.flatten(end_dim=1)
        return {
            'edges_neg': torch.index_select(output, 0, edges_neg[:, 0]) * torch.index_select(output, 0, edges_neg[:, 1]),
            'edges_pos': torch.index_select(output, 0, edges_pos[:, 0]) * torch.index_select(output, 0, edges_pos[:, 1])}

    def loss(prediction, data, coef_neg=1., weight_types=None):
        loss_edge_pos = torch.mean(torch.nn.functional.softplus(-prediction['edges_pos']))
        loss_edge_neg = torch.mean(torch.nn.functional.softplus(prediction['edges_neg']))

        loss_type = torch.nn.functional.cross_entropy(prediction['type'], data.edges_toInf_pos_types.cuda(), weight=weight_types)

        return loss_edge_pos + coef_neg * loss_edge_neg + loss_type

    def performances(prediction, data):
        """
        To assess the precision and the recall of the neural network.
        """
        with torch.no_grad():
            n_edges_pos_predicted_pos = torch.sum(prediction['edges_pos'] > 0).item()
            n_edges_predicted_pos = n_edges_pos_predicted_pos + torch.sum(prediction['edges_neg'] > 0).item()
            n_edges_pos = len(prediction['edges_pos'])

            types_evaluated_i = torch.arange(len(EDGE_IDX_MAP)).unsqueeze(0).to(data.edges_toInf_pos_types.device).cuda()
            data.edges_toInf_pos_types = data.edges_toInf_pos_types.unsqueeze(1)

            i_predicted = torch.argmax(prediction['type'], dim=-1, keepdim=True)
            n_edges_i_predicted_i = torch.count_nonzero((i_predicted == types_evaluated_i) & (i_predicted == data.edges_toInf_pos_types.cuda()), axis=0)
            n_edges_predicted_i = torch.count_nonzero(i_predicted == types_evaluated_i, axis=0)
            n_edges_i = torch.count_nonzero(data.edges_toInf_pos_types.cuda() == types_evaluated_i, axis=0)

        return ([n_edges_pos_predicted_pos, n_edges_predicted_pos, n_edges_pos],
                [n_edges_i_predicted_i.tolist(), n_edges_predicted_i.tolist(), n_edges_i.tolist()])

