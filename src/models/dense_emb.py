import torch
from typing import Dict
from .concatenatelinear import ConcatenateLinear

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

class DenseSparsePreEmbedding(torch.nn.Module):
    """
        A torch module that mixes embeddings for edges or nodes.
    """

    def __init__(self, feature_embeddings, fixed_embedding_cardinality: int = None, fixed_embedding_dim: int = None,
                 sparse_embedding_dim: int = None, embedding_dim: int = None, padding_idx: int = None):
        """
            feature_embeddings (Dict/iterable) : a dict containing the feature embeddings
        """
        super(DenseSparsePreEmbedding, self).__init__()
        self.feature_embeddings = torch.nn.ModuleDict(feature_embeddings)  # An ordered Dict constructed from the dict of embeddings
        self.fixed_embedding = torch.nn.Embedding(fixed_embedding_cardinality, fixed_embedding_dim, padding_idx=padding_idx)  # A simple lookup table that stores embeddings of a fixed dictionary and size.

        embedding_dim = embedding_dim or fixed_embedding_dim
        self.sparse_embedding_dim = sparse_embedding_dim or fixed_embedding_dim
        self.dense_merge = ConcatenateLinear(fixed_embedding_dim, self.sparse_embedding_dim, embedding_dim)

    def forward(self, fixed_features: torch.Tensor, sparse_features_index: Dict[str, torch.Tensor], sparse_features_value : Dict[str, torch.Tensor]):
        """
        fixed_features: tensor, the identifiers of the types of the primitives/constraints;
        sparse_features: dictionary {primitive/constraint: {'index': tensor, 'value': tensor}}, the discretized parameters of the primitives/constraints.
        """

        fixed_embeddings = self.fixed_embedding(fixed_features)
        sparse_embeddings = self.generate_sparse_embeddings(fixed_embeddings, sparse_features_index, sparse_features_value)


        return self.dense_merge(fixed_embeddings, sparse_embeddings)


    def generate_sparse_embeddings(self, fixed_embeddings: torch.Tensor, sparse_features_index: Dict[str, torch.Tensor], sparse_features_value : Dict[str, torch.Tensor]):

        """
            This function computes sparse embeddings from embeddings and features.
            fixed_embeddings (torch.tensor) :
            sparse_features (Dict) :
        """

        sparse_embeddings = fixed_embeddings.new_zeros((fixed_embeddings.shape[0], self.sparse_embedding_dim))

        # Filter on the sparse embedding matrix
        for k, embedding_network in self.feature_embeddings.items():
            # logger.info(f'sparser {sparse_features}')
            if k in sparse_features_index :
                sf_index = sparse_features_index[k]
                if len(sf_index) != 0: # Si le type n'est pas présent, on continue
                    sf_value = sparse_features_value[k]
                    # assert (sf['index'] < fixed_embeddings.shape[0]).all()
                    sparse_embeddings[sf_index] = embedding_network(sf_value) # met à jour la ligne avec la valeur associée
        return sparse_embeddings
