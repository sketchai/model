import torch
from typing import Dict

from src.models.numerical_features.embedding import NumericalFeaturesEmbedding
from src.models.numerical_features.encoding import NumericalFeatureEncoding


def generate_embedding(d_features_dims: Dict = {}, embedding_dim: int = None) -> Dict:
    """
        This function generates a dict of embeddings
    """
    d_embedding = {}
    for k, elt in d_features_dims.items():
        d_embedding[k.name] = torch.nn.Sequential(NumericalFeatureEncoding(elt.values(), embedding_dim),
                                                  NumericalFeaturesEmbedding())
    return d_embedding
