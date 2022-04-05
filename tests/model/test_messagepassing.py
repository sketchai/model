import unittest
import logging
import torch
import pickle
from src.utils.to_dict import yaml_to_dict
import sys 
sys.path.append('/home/i37181/Documents/Projets/CAO/SketchGraphs/sketchgraphs')


from src.models.dense_emb import DenseSparsePreEmbedding

from src.models.gat import GaT
from src.dataloader.generate_dataModule import SketchGraphDataModule



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class TestMessagePassing(unittest.TestCase):

    def setUp(self):
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

        logger.debug(f'NODE_IDX_MAP={NODE_IDX_MAP}')

        # logger.info(f'--- d_prep= {d_prep}')
        graph_dataset = SketchGraphDataModule(conf, d_prep)
        self.dataset = graph_dataset.train_dataloader()
        logger.debug(f'dataset size={len(self.dataset)}')

        # Model initialization
        d_model = conf.get('model')

        use_cuda = not d_model.get('cpu') and torch.cuda.is_available()
        self.device = torch.device('cuda') if use_cuda else 'cpu'

        self.gat = GaT(d_model, d_prep)
        self.gat.to(self.device)

    def test_message(self):


        for i, data in enumerate(self.dataset):
            if i == 0 :
                continue
            # Compute node and edge embedding
            logger.info(f'data.node_features: {data.node_features}')
            logger.info(f'data.sparse_node_features: {data.sparse_node_features}')

            node_embedding = self.gat.node_embedding(data.node_features, data.sparse_node_features)
            edge_embedding = self.gat.edge_embedding(data.edge_features, data.sparse_edge_features)

            # Agregate node and edge information (message passing)
            logger.info(f'node embedding: {node_embedding}')
            logger.info(f'edge embedding: {edge_embedding}')
            agreg = self.gat.aggregate_by_incidence(node_embedding, data.incidences.to(self.device), edge_embedding)

            logger.info(f'agreg: {agreg}')
            # input_embedding = node_embedding + agreg
            break



