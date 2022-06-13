import os
import unittest
import logging
import pickle
from sketch_gnn.utils.to_dict import parse_config



from sketch_gnn.models.dense_emb import DenseSparsePreEmbedding

from sketch_gnn.models.numerical_features.generator import generate_embedding
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule



logger = logging.getLogger(__name__)


class TestDenseSparseEmbedding(unittest.TestCase):



    def test_forward(self):
        # Load an example
        conf = parse_config('tests/asset/mock/gat_example.yml')            
        with open(conf.get('file_prep_parms'), 'rb') as f:
            d_prep = pickle.load(f)

        # logger.info(f'--- d_prep= {d_prep}')
        graph_dataset = SketchGraphDataModule(conf, d_prep)
        dataset = graph_dataset.train_dataloader()
        logger.debug(f'dataset size={len(dataset)}')
        logger.debug(f'dataset.dataset: nb size={len(dataset.dataset)}')

        embedding_dim = 240
        node_feature = generate_embedding(d_prep.get('node_feature_dimensions'), embedding_dim)
        logger.debug(f'node_feature: {node_feature}')
        node_embedding = DenseSparsePreEmbedding(feature_embeddings= node_feature, 
                                                        fixed_embedding_cardinality=len(d_prep.get('node_idx_map')), 
                                                        fixed_embedding_dim= embedding_dim, 
                                                        padding_idx=d_prep.get('padding_idx'))

        # device = torch.device('cuda') 

        for i, batch in enumerate(dataset):
            class AttrDict(dict):
                def __init__(self, base_dict:dict):
                    self.__dict__ = base_dict

            # batch : 2 circles
            batch = AttrDict(batch)
            logger.info(f'Fixe feature')
            logger.info(f'Input tensor keys: {batch.node_features}')
            fixed_embeddings = node_embedding.fixed_embedding(batch.node_features)
            # output = node_embedding.forward(batch.node_features, batch.sparse_node_features)
            logger.debug(f'Output tensor: {fixed_embeddings}')

            logger.info(f'Sparse features: { batch.sparse_node_features}')
            output = node_embedding.generate_sparse_embeddings(fixed_embeddings, batch.sparse_node_features)
            
            # logger.debug(f'Output tensor: {output}')
            logger.debug(f'Output tensor: {output.sum()}')
            circle_1 = output[1]
            logger.debug(f'Circle 1: {circle_1.sum()}')
            self.assertEqual(len(circle_1), embedding_dim)
            self.assertTrue(abs(circle_1.sum()) > 0.0)

            circle_2 = output[3]
            logger.debug(f'Circle 2: {circle_2.sum()}')
            self.assertTrue(abs(circle_2.sum()) > 0.0)


            break
        # TODO : Bug au niveau du feature embedding
    


        
