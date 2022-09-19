import torch
import logging
from typing import Dict

from sketch_gnn.models.numerical_features.merge import NumericalFeaturesMerge
from sketch_gnn.models.numerical_features.embedding import NumericalFeatureEmbedding

logger = logging.getLogger(__name__)

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
    logger.debug(d_features_dims)
    n_max_params = max(len(params) for params in d_features_dims.values())
    d_embedding = {}
    for k, elt in d_features_dims.items():
        d_embedding[k] = torch.nn.Sequential(NumericalFeatureEmbedding(elt.values(), embedding_dim, n_max_params),
                                                  NumericalFeaturesMerge())
    return d_embedding
