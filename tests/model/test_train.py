import unittest
import os
import pickle
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sketch_gnn.models.predict import PredictSketch
import logging
from sketch_gnn.utils.to_dict import parse_config
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule
from sketch_gnn.models.gat import GaT
logger = logging.getLogger(__name__)

class TestTrainingLoop(unittest.TestCase):
    def test_training_loop(self):
        conf = parse_config('tests/asset/mock/gat_v2.yml')
        # Initialize parameters
        d_train = conf.get('train')
        with open(conf.get('prep_parms_path'), 'rb') as f:
            preprocessing_params = pickle.load(f)
        # Create DataLoader
        data = SketchGraphDataModule(conf)

        ######## STEP 2 : Init Model
        logger.info('-- Model initialization:...')
        # Model initialization
        d_model = conf.get('model')
        model = GaT(d_model, preprocessing_params)

        conf['edge_idx_map'] = preprocessing_params.get('edge_idx_map')
        conf['node_idx_map'] = preprocessing_params.get('node_idx_map')


        sketchPredictionmodel = PredictSketch(
            model= model,
            conf= conf)
        logger.info('-- Model initialization: end')

        logger.info('-- Logger and Trainer initialization:...')
        logger_conf = conf.get('logger')

        logger_tensorboard = TensorBoardLogger(
            save_dir = logger_conf.get('save_dir'),
            name = logger_conf.get('name'),
            log_graph=False,
            default_hp_metric=False)
        ######## STEP 3 : Init Trainer and launch training
        trainer = pl.Trainer(
            accelerator='cpu',
            callbacks=[],
            max_epochs=2,
            logger=logger_tensorboard,
            )
        results = trainer.fit(sketchPredictionmodel, data)
    