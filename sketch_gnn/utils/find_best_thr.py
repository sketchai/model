"""Find best threshold on validation set"""

import logging
import numpy as np 
import pytorch_lightning as pl
import pickle
from argparse import ArgumentParser
from sketch_gnn.utils.logger import logger
from sketch_gnn.utils.to_dict import parse_config
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule
from sketch_gnn.models.gat import GaT
from sketch_gnn.models.predict import PredictSketch
from sketch_gnn.inference.metrics import sketch_wise_precision_recall

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', help= 'path to your model checkpoint')

    args = parser.parse_args()
    ######## STEP 1 : Init Datasets
    conf = parse_config('config/gat.yml')
    # Initialize parameters
    d_train = conf.get('train')
    with open(conf.get('prep_parms_path'), 'rb') as f:
        preprocessing_params = pickle.load(f)
    # Create DataLoader
    data = SketchGraphDataModule(conf,preprocessing_params)

    ######## STEP 2 : Init Model
    logger.info('-- Model initialization:...')
    # Model initialization
    d_model = conf.get('model')
    model = GaT(d_model, preprocessing_params)

    conf['edge_idx_map'] = preprocessing_params.get('edge_idx_map')
    conf['node_idx_map'] = preprocessing_params.get('node_idx_map')

    # args.path = '~/data/sg/models/v004/gat-epoch=102-val_loss=0.00.ckpt'

    sketchPredictionmodel = PredictSketch.load_from_checkpoint(
        checkpoint_path=args.path,
        model= model,
        conf= conf)
    logger.info('-- Model initialization: end')

    logger.info('-- Logger and Trainer initialization:...')
    ######## STEP 3 : Init Trainer and launch evaluation on test
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[],
        logger=False,
        )
    _ = trainer.test(sketchPredictionmodel, dataloaders=data.val_dataloader())
    results = sketchPredictionmodel.test_results

    best_thr = 0
    max_f1_score = 0
    for thr in np.linspace(0.90,0.99,10):
        avg_precision, avg_recall = sketch_wise_precision_recall(results, thr=thr)
        f1_score = (avg_precision*avg_recall)/(avg_precision+avg_recall)
        print(f'thr = {thr:0.3}, avg precision = {avg_precision:0.3}, avg_recall = {avg_recall:0.3}, f1_score ={f1_score:0.3}')
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            best_thr = thr
    print(f'Best threshold = {best_thr}')
    