import torch
from typing import Dict
import pytorch_lightning as pl

from sketch_gnn.models.numerical_features.generator import generate_embedding
from .concatenatelinear import ConcatenateLinear

import logging
logger = logging.getLogger(__name__)

class NodeEmbeddingLayer(pl.LightningModule):
    """
        A torch module that mixes embeddings for class label and parameters.
    """

    def __init__(self,
        feature_dims,
        embedding_dim: int,
        node_idx_map: dict,
        ):
        """
        """
        super().__init__()
        self.node_idx_map = node_idx_map
        if 'void' in self.node_idx_map:
            del self.node_idx_map['void']
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        
        sparse_embedding_layers = generate_embedding(feature_dims, embedding_dim)
        self.sparse_embedding_layers = torch.nn.ModuleDict(sparse_embedding_layers)
        embedding_cardinality = len(self.node_idx_map)
        self.fixed_embedding_layer = torch.nn.Embedding(embedding_cardinality, embedding_dim)
        self.dense_merge = ConcatenateLinear(sizes=[embedding_dim, embedding_dim],output_size=embedding_dim)

    def forward(self, node_features) -> torch.tensor:
        """
        node_features: torch tensor

        first column must be an int representing the primitive's label (e.g. LINE, POINT etc.)
        """

        fixed_features = node_features[:,0]
        sparse_features = node_features[:,1:]
        fixed_embeddings = self.fixed_embedding_layer(fixed_features)

        sparse_embeddings = fixed_embeddings.new_zeros((fixed_embeddings.shape[0], self.embedding_dim))

        for k, idx in self.node_idx_map.items():
            if 'SN_' in k:
                embedding_network = self.sparse_embedding_layers['POINT']
            else:
                embedding_network = self.sparse_embedding_layers[k]
            mask = fixed_features==idx
            sparse_embeddings[mask] = embedding_network(sparse_features[mask])

        return self.dense_merge(fixed_embeddings, sparse_embeddings)
