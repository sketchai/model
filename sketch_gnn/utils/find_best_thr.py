"""Find best threshold on validation set"""

import logging
import numpy as np 
import pytorch_lightning as pl
import pickle
from argparse import ArgumentParser

from torch import threshold
from sketch_gnn.utils.to_dict import parse_config
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule
from sketch_gnn.models.gat import GaT
from sketch_gnn.models.predict import PredictSketch
from sketch_gnn.inference.metrics import sketch_wise_precision_recall
logger = logging.getLogger(__name__)

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', help= 'path to your model checkpoint')
    parser.add_argument('--test', default=False, type=bool, help= 'if true, also run test')

    args = parser.parse_args()
    ######## STEP 1 : Init Datasets
    conf = parse_config('config/gat.yml')
    # Initialize parameters
    d_train = conf.get('train')
    with open(conf.get('prep_parms_path'), 'rb') as f:
        preprocessing_params = pickle.load(f)


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
    thresholds = []
    props = [0., 0.5, 0.75, 0.9]
    for prop in props:
        conf['val_data']['prop_max_edges_given'] = prop
        print('-'*10 + f'Given edges: {int(100*prop)}%')
        # Create DataLoader
        data = SketchGraphDataModule(conf,preprocessing_params)
        _ = trainer.test(sketchPredictionmodel, dataloaders=data.val_dataloader())
        results = sketchPredictionmodel.test_results

        best_thr = 0
        max_f1_score = 0
        for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]:
            avg_precision, avg_recall = sketch_wise_precision_recall(results, thr=thr)
            f1_score = (avg_precision*avg_recall)/(avg_precision+avg_recall)
            print(f'thr = {thr:0.3}, avg precision = {avg_precision:0.3}, avg_recall = {avg_recall:0.3}, f1_score ={f1_score:0.3}')
            if f1_score > max_f1_score:
                max_f1_score = f1_score
                max_avg_precision = avg_precision
                max_avg_recall = avg_recall
                best_thr = thr
        print(f'Best threshold = {best_thr:0.2}, precision {max_avg_precision:0.2}, recall {max_avg_recall:0.2}')
        thresholds.append(best_thr)
    
    if args.test:
        for prop, thr in zip(props, thresholds):
            conf['test_data']['prop_max_edges_given'] = prop
            data = SketchGraphDataModule(conf,preprocessing_params)
            _ = trainer.test(sketchPredictionmodel, dataloaders=data.test_dataloader())
            results = sketchPredictionmodel.test_results
            avg_precision, avg_recall = sketch_wise_precision_recall(results, thr=thr)
            print(f'Prop hidden: {prop:0.2}, avg precision = {avg_precision:0.2}, avg_recall = {avg_recall:0.2}')


    
    