import unittest
import logging
import torch
import pickle
from src.utils.to_dict import yaml_to_dict
import sys 
sys.path.append('/home/i37181/Documents/Projets/CAO/SketchGraphs/sketchgraphs')


from src.models.dense_emb import DenseSparsePreEmbedding

from src.models.numerical_features.generator import generate_embedding
from src.dataloader.generate_dataModule import SketchGraphDataModule



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class TestDenseSparseEmbedding(unittest.TestCase):



    def test_forward(self):
        # Load an example
        conf = yaml_to_dict('tests/asset/mock/gat_example.yml')
        d_train = conf.get('train')
        with open(d_train.get('prep_parms_path'), 'rb') as f:
            d_prep = pickle.load(f)
        # Add node_idx_map and edge_idx_map (must be placed directly into the preprocessing files)
        from src.utils.maps import NODE_IDX_MAP, EDGE_IDX_MAP, PADDING_IDX
        d_prep['node_idx_map'] = NODE_IDX_MAP
        d_prep['edge_idx_map'] = EDGE_IDX_MAP
        d_prep['padding_idx'] = PADDING_IDX

        # logger.info(f'--- d_prep= {d_prep}')
        graph_dataset = SketchGraphDataModule(conf, d_prep)
        dataset = graph_dataset.train_dataloader()

        embedding_dim = 240
        node_feature = generate_embedding(d_prep.get('node_feature_dimensions'), embedding_dim)
        node_embedding = DenseSparsePreEmbedding(feature_embeddings= node_feature, 
                                                        fixed_embedding_cardinality=len(d_prep.get('node_idx_map')), 
                                                        fixed_embedding_dim= embedding_dim, 
                                                        padding_idx=d_prep.get('padding_idx'))


        for i, batch in enumerate(dataset):
            logger.info(f'Input tensor: \n')
            logger.info(dataset)
            output = node_embedding.forward(batch)
            logger.debug(f'Output tensor: {output}')

            break
        # TODO : Tester cette couche pour verifier la forme de la matrice (notamment, est-elle totalement écrasée par la couche d'avant ? oui/non)