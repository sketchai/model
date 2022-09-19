import torch
import pytorch_lightning as pl
import logging
logger = logging.getLogger(__name__)


class NumericalFeatureEmbedding(pl.LightningModule):
    """Encode an array of numerical features.

        This encodes a sequence of features (presented as a sequence of integers)
        into a sequence of vector through an embedding.
    ----------------------------
    Source : SketchGraph model. sketchgraphs_models/graph/model/numerical_features.py
    """

    def __init__(self, feature_dims, embedding_dim, n_max_params):
        super(NumericalFeatureEmbedding, self).__init__()
        feature_dims = list(feature_dims)
        feature_dims_padded = feature_dims + (n_max_params-len(feature_dims))*[0]
        self.register_buffer(
            'feature_offsets',
            torch.cumsum(torch.tensor([0] + feature_dims_padded[:-1], dtype=torch.int64), dim=0))  # Save feature_offsets into the gpu

        # logger.debug(f'feature dim : {self.feature_offsets}')  # Decalage des paramètres 
        cardinality_embeddings = sum(feature_dims)
        self.embeddings = torch.nn.Embedding(cardinality_embeddings + 1, embedding_dim, sparse=False, padding_idx=cardinality_embeddings)

    def forward(self, features):
        y = self.embeddings(features + self.feature_offsets)
        return y
