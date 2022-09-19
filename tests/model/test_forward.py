import os
import unittest
import logging
import torch
import pickle
from sketch_gnn.utils.to_dict import parse_config

from sketch_gnn.models.gat import GaT
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule

logger = logging.getLogger(__name__)


class TestForward(unittest.TestCase):

    def setUp(self):
        # Load an example
        conf = parse_config('tests/asset/mock/gat_example.yml')
        with open(conf.get('file_prep_parms'), 'rb') as f:
            d_prep = pickle.load(f)

        # logger.info(f'--- d_prep= {d_prep}')
        graph_dataset = SketchGraphDataModule(conf)
        self.dataset = graph_dataset.train_dataloader()

        # Model initialization
        d_model = conf.get('model')

        use_cuda = not d_model.get('cpu') and torch.cuda.is_available()
        self.device = torch.device('cuda') if use_cuda else 'cpu'

        self.gat = GaT(d_model, d_prep)
        self.gat.to(self.device)

    def test_forward(self):

        for i, data in enumerate(self.dataset):
            # Compute node and edge embedding
            logger.debug(data)
            output = self.gat(data)
            logger.debug(output)
            break
        for k, param in self.gat.named_parameters():
            logger.debug(f'{k} : {param.shape}')



