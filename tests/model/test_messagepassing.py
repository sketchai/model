import os
import unittest
import logging
import torch
import pickle
from sketch_gnn.utils.to_dict import parse_config



from sketch_gnn.models.dense_emb import DenseSparsePreEmbedding

from sketch_gnn.models.gat import GaT
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule



logger = logging.getLogger(__name__)


class TestMessagePassing(unittest.TestCase):

    def setUp(self):
        # Load an example
        conf = parse_config('tests/asset/mock/gat_example.yml')
        with open(conf.get('file_prep_parms'), 'rb') as f:
            d_prep = pickle.load(f)

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



