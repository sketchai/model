import numpy as np
import torch

from sketchgraphs_models.graph.model import numerical_features

from models.dense_emb import DenseSparsePreEmbedding, ConcatenateLinear
from maps.maps import *


class GravTransformer(torch.nn.Module):
    """
    The neural network. Some utilitaries are included.
    """
    def __init__(self, node_feature_dimensions, edge_feature_dimensions, embedding_dim, n_head, num_layers, do_positional_encoding, lMax):
        """
        node_feature_dims: dictionary {primitive: {feature: dimension}}, as returned by the preprocessing in 'preprocessing_params.pkl';
        edge_feature_dims: dictionary {constraint: {feature: dimension}}, as returned by the preprocessing in 'preprocessing_params.pkl';
        embedding_dim: int, the embedding dimension used in the whole network;
        n_head: int, number of heads in the multi-head attention mecanism;
        num_layers: int, number of layers for the encoder;
        do_positional_encoding: bool, set to True for adding a positional encoding;
        lMax: int, length of the examples, returned by the preprocessing in 'preprocessing_params.pkl'.
        """
        super(GravTransformer, self).__init__()
        self.lMax = lMax
        self.embedding_dim = embedding_dim
        self.node_embedding = DenseSparsePreEmbedding(
            {
                k.name: torch.nn.Sequential(
                    numerical_features.NumericalFeatureEncoding(fd.values(), embedding_dim),
                    numerical_features.NumericalFeaturesEmbedding(embedding_dim)
                )
                for k, fd in node_feature_dimensions.items()
            },
            len(NODE_IDX_MAP), embedding_dim, padding_idx=PADDING_IDX)
        self.edge_embedding = DenseSparsePreEmbedding(
            {
                k.name: torch.nn.Sequential(
                    numerical_features.NumericalFeatureEncoding(fd.values(), embedding_dim),
                    numerical_features.NumericalFeaturesEmbedding(embedding_dim)
                )
                for k, fd in edge_feature_dimensions.items()
            },
            len(EDGE_IDX_MAP), embedding_dim)
        self.transform_edge_messages = ConcatenateLinear(embedding_dim, embedding_dim, embedding_dim)
        self.do_positional_encoding = do_positional_encoding
        if do_positional_encoding:
            self.positional_encoding = torch.nn.Embedding(lMax, embedding_dim)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, dim_feedforward=2*embedding_dim)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.prediction_edge = torch.nn.Linear(embedding_dim, 1)
        self.prediction_type = torch.nn.Linear(embedding_dim, len(EDGE_IDX_MAP))
        
    def forward(self, data):
        node_embedding = self.node_embedding(data.node_features, data.sparse_node_features)
        edge_embedding = self.edge_embedding(data.edge_features, data.sparse_edge_features)
        
        agreg = self.aggregate_by_incidence(node_embedding, data.incidences, edge_embedding)
        input_embedding = node_embedding + agreg
        
        if self.do_positional_encoding:
            input_embedding += self.positional_encoding(data.positions.tile(data.l_batch))
            
        input_embedding = input_embedding.view((data.l_batch, self.lMax, self.embedding_dim))
        output_transformer = torch.transpose(self.transformer_encoder(torch.transpose(input_embedding, 0,1),
                                                                      src_key_padding_mask=data.src_key_padding_mask), 0,1)
        #output_transformer = input_embedding
        
        edges_neg, edges_pos = data.edges_toInf_neg, data.edges_toInf_pos
        representation_final_edges = GravTransformer.representation_final_edges(output_transformer, edges_neg, edges_pos)
        
        prediction_edge_pos = self.prediction_edge(representation_final_edges['edges_pos'])
        prediction_edge_neg = self.prediction_edge(representation_final_edges['edges_neg'])
        prediction_type = self.prediction_type(representation_final_edges['edges_pos'])
        
        return {"edges_pos": prediction_edge_pos,
                "edges_neg": prediction_edge_neg,
                "type": prediction_type}
        
    def aggregate_by_incidence(self, node_embedding, incidence, edge_embedding):
        edge_messages = node_embedding.index_select(0, incidence[1])
        edge_messages = self.transform_edge_messages(edge_messages, edge_embedding)
        
        output = node_embedding.new_zeros([node_embedding.shape[0]] + list(edge_messages.shape[1:]))
        output.index_add_(0, incidence[0], edge_messages)        
        return output
    
    def representation_final_edges(output, edges_neg, edges_pos):
        output = output.flatten(end_dim=1)
        return {
            'edges_neg': torch.index_select(output, 0, edges_neg[:,0]) * torch.index_select(output, 0, edges_neg[:,1]),
            'edges_pos': torch.index_select(output, 0, edges_pos[:,0]) * torch.index_select(output, 0, edges_pos[:,1])}
    
    def loss(prediction, data, coef_neg=1., weight_types=None):
        loss_edge_pos = torch.mean(torch.nn.functional.softplus(-prediction['edges_pos']))
        loss_edge_neg = torch.mean(torch.nn.functional.softplus(prediction['edges_neg']))

        loss_type = torch.nn.functional.cross_entropy(prediction['type'], data.edges_toInf_pos_types, weight=weight_types)
        
        return loss_edge_pos + coef_neg*loss_edge_neg + loss_type
    
    def performances(prediction, data):
        """
        To assess the precision and the recall of the neural network.
        """
        with torch.no_grad():
            n_edges_pos_predicted_pos = torch.sum(prediction['edges_pos'] > 0).item()
            n_edges_predicted_pos = n_edges_pos_predicted_pos + torch.sum(prediction['edges_neg'] > 0).item()
            n_edges_pos = len(prediction['edges_pos'])
            
            types_evaluated_i = torch.arange(len(EDGE_IDX_MAP)).unsqueeze(0).to(data.edges_toInf_pos_types.device)
            data.edges_toInf_pos_types = data.edges_toInf_pos_types.unsqueeze(1)

            i_predicted = torch.argmax(prediction['type'], dim=-1, keepdim=True)
            n_edges_i_predicted_i = torch.count_nonzero((i_predicted == types_evaluated_i) & (i_predicted == data.edges_toInf_pos_types), axis=0)
            n_edges_predicted_i = torch.count_nonzero(i_predicted == types_evaluated_i, axis=0)
            n_edges_i = torch.count_nonzero(data.edges_toInf_pos_types == types_evaluated_i, axis=0)
            
        return ([n_edges_pos_predicted_pos, n_edges_predicted_pos, n_edges_pos],
            [n_edges_i_predicted_i.tolist(), n_edges_predicted_i.tolist(), n_edges_i.tolist()])
        
    
