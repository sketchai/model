import unittest
import os
import pickle
import numpy as np
import torch
import logging
from sketch_gnn.utils.to_dict import parse_config
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule
from sketch_gnn.models.gat import GaT
from sketch_gnn.inference.eval import EvalPrediction

logger = logging.getLogger(__name__)
class TestEvalPrediction(unittest.TestCase):
    def setUp(self):
        
        # Load config (.yml)
        conf = parse_config('tests/asset/mock/gat_example.yml')
        with open(conf.get('prep_parms_path'), 'rb') as f:
            d_prep = pickle.load(f)
        self.edge_idx_map = d_prep.get('edge_idx_map')

        # Load dataset
        graph_dataset = SketchGraphDataModule(conf, d_prep)
        train_dataset = graph_dataset.train_dataloader()
        data_gen = iter(train_dataset)

        # Load model (without weights)
        d_model = conf.get('model')
        gat = GaT(d_model, d_prep)

        # Inference
        self.batch = next(data_gen)
        self.output = gat(self.batch)
        
        self.pred = EvalPrediction(
            batch=self.batch,
            output=self.output,
            edge_idx_map=self.edge_idx_map,
            threshold=0.8,
            )
        
        self.pred.print_prediction()

    def test_getitem(self):

        expected_keys = [
            'category',
            'predicted_sigmoid',
            'true_label',
            'predicted_type_name',
            'true_type_name',
            'references']

        n_edges = min(10, len(self.pred))
        for i in range(n_edges):
            info = self.pred[i]
            self.assertIsInstance(info,dict)
            for key in info.keys():
                logger.debug(info)
                self.assertTrue(key in expected_keys)

    def test_init(self):
        n_edges_toInf_pos = self.batch['edges_toInf_pos'].shape[0]
        n_edges_toInf_neg = self.batch['edges_toInf_neg'].shape[0]
        n_edges_given = self.batch['incidences'].shape[1]//2
        n_edges_toInf = n_edges_toInf_neg + n_edges_toInf_pos
        n_edges_total = n_edges_toInf + n_edges_given

        array_shape = {
            'predicted_sigmoid': (n_edges_toInf,),
            'true_label': (n_edges_toInf,),
            'references': (n_edges_total,2),
        }

        array_dtype = {
            'predicted_sigmoid': np.float32,
            'true_label': np.float,
            'references': np.int,
        }

        for name, shape in array_shape.items():
            array = self.pred.__dict__.get(name)
            self.assertEqual(shape, array.shape)

        for name, dtype in array_dtype.items():
            array = self.pred.__dict__.get(name)
            self.assertEqual(dtype, array.dtype)

        list_len = {
            'predicted_type_name': n_edges_toInf,
            'true_type_name': n_edges_total,
        }

        for name, len_ in list_len.items():
            list_ = self.pred.__dict__.get(name)
            self.assertIsInstance(list_, list)
            self.assertEqual(len_, len(list_))