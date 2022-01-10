import torch

class ConcatenateLinear(torch.nn.Module):
    """
    A torch module which concatenates several inputs and mixes them using a linear layer.
    """
    def __init__(self, left_size, right_size, output_size):
        super(ConcatenateLinear, self).__init__()
        self.left_size = left_size
        self.right_size = right_size
        self.output_size = output_size

        self._linear = torch.nn.Linear(left_size + right_size, output_size)

    def forward(self, left, right):
        return self._linear(torch.cat((left, right), dim=-1))

class DenseSparsePreEmbedding(torch.nn.Module):
    """
    A torch module that mixes embeddings for edges or nodes.
    """
    def __init__(self, feature_embeddings, fixed_embedding_cardinality, fixed_embedding_dim,
                 sparse_embedding_dim=None, embedding_dim=None, padding_idx=None):
        super(DenseSparsePreEmbedding, self).__init__()
        sparse_embedding_dim = sparse_embedding_dim or fixed_embedding_dim
        embedding_dim = embedding_dim or fixed_embedding_dim

        self.feature_embeddings = torch.nn.ModuleDict(feature_embeddings)
        self.sparse_embedding_dim = sparse_embedding_dim
        self.fixed_embedding_dim = fixed_embedding_dim
        self.fixed_embedding = torch.nn.Embedding(fixed_embedding_cardinality, fixed_embedding_dim, padding_idx=padding_idx)
        self.dense_merge = ConcatenateLinear(fixed_embedding_dim, sparse_embedding_dim, embedding_dim)

    def forward(self, fixed_features, sparse_features):
        """
        fixed_features: tensor, the identifiers of the types of the primitives/constraints;
        sparse_features: dictionary {primitive/constraint: {'index': tensor, 'value': tensor}}, the discretized parameters of the primitives/constraints.
        """
        fixed_embeddings = self.fixed_embedding(fixed_features)
        sparse_embeddings = fixed_embeddings.new_zeros((fixed_embeddings.shape[0], self.sparse_embedding_dim))

        for k, embedding_network in self.feature_embeddings.items():
            sf = sparse_features.get(k)
            if sf is None or len(sf['index']) == 0:
                continue

            assert (sf['index'] < fixed_embeddings.shape[0]).all()
            sparse_embeddings[sf['index']] = embedding_network(sf['value'])

        return self.dense_merge(fixed_embeddings, sparse_embeddings)
