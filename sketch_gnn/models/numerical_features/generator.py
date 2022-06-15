import torch
from typing import Dict

from sketch_gnn.models.numerical_features.embedding import NumericalFeaturesEmbedding
from sketch_gnn.models.numerical_features.encoding import NumericalFeatureEncoding


def generate_embedding(d_features_dims: Dict = {}, embedding_dim: int = None) -> Dict:
    """
        This function generates a dict of embeddings.
        Inputs  :
            d_features_dims (Dict) : a dict { elt : {component_1 : int, component_2 : int, ...}}
            embedding_dim (int)    : a integer representing the final size of the embedding vector
        Outputs :
            d_embedding (Dict) : a dict {elt.name : torch.nn.Module} where the torch.nn.Module encodes the feature into a vector of size embedding_dim
        ------------------------------------------------------------------------------------------

        Specific carateristics:
            - d_features_dims.keys() elements must have a .name attribut
    """
    d_embedding = {}
    for k, elt in d_features_dims.items():
        d_embedding[k] = torch.nn.Sequential(NumericalFeatureEncoding(elt.values(), embedding_dim),
                                                  NumericalFeaturesEmbedding())
    return d_embedding
