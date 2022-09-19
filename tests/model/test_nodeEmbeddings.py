import os
import unittest
import logging
import pickle
from sketch_gnn.utils.to_dict import parse_config



from sketch_gnn.models.node_embedding import NodeEmbeddingLayer

from sketch_gnn.models.numerical_features.generator import generate_embedding
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule
logger = logging.getLogger(__name__)

class TestNodeEmbeddings(unittest.TestCase):

    def test_forward(self):
        # Load an example
        conf = parse_config('tests/asset/mock/gat_example.yml')            
        with open(conf.get('file_prep_parms'), 'rb') as f:
            d_prep = pickle.load(f)

        graph_dataset = SketchGraphDataModule(conf)
        dataloader = graph_dataset.train_dataloader()

        embedding_dim = 4
        node_embedding = NodeEmbeddingLayer(feature_dims= d_prep.get('node_feature_dimensions'), 
                                        embedding_dim= embedding_dim,
                                        node_idx_map= d_prep.get('node_idx_map')
        )
        iterator = iter(dataloader)
        batch = next(iterator)
        logger.debug(f'batch {batch}')
        logger.debug(f'batch.x_p.shape {batch.x.shape}')

        out = node_embedding.forward(node_features=batch.x)
        logger.debug(f'out.shape {out}')
        


        
