import torch

import logging
logger = logging.getLogger(__name__)


class NumericalFeatureEncoding(torch.nn.Module):
    """Encode an array of numerical features.

        This encodes a sequence of features (presented as a sequence of integers)
        into a sequence of vector through an embedding.
    ----------------------------
    Source : SketchGraph model. sketchgraphs_models/graph/model/numerical_features.py
    """

    def __init__(self, feature_dims, embedding_dim):
        super(NumericalFeatureEncoding, self).__init__()
        feature_dims = list(feature_dims)
        self.register_buffer(
            'feature_offsets',
            torch.cumsum(torch.tensor([0] + feature_dims[:-1], dtype=torch.int64), dim=0))  # Save feature_offsets into the gpu

        # logger.debug(f'feature dim : {self.feature_offsets}')  # Decalage des paramètres 

        self.embeddings = torch.nn.Embedding(sum(feature_dims), embedding_dim, sparse=False)

    def forward(self, features):
        y = self.embeddings(features.to(self.feature_offsets.device) + self.feature_offsets)
        return y
