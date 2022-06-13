"""
Evaluate model on test set

Threshold value can be adjusted first on validation set by running:
python sketch_gnn/utils/find_best_thr.py --path path/to/your model
"""

import logging 
import pytorch_lightning as pl
import pickle
import numpy as np
from argparse import ArgumentParser
from sketch_gnn.utils.to_dict import parse_config
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule
from sketch_gnn.models.gat import GaT
from sketch_gnn.models.predict import PredictSketch
from sketch_gnn.inference.metrics import sketch_wise_precision_recall

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = ArgumentParser()
    parser.add_argument('--path', help= 'path to your model checkpoint')
    parser.add_argument('--thr', default=0.98, help='threshold used for evaluation', type=float)
    parser.add_argument('--prop', help='prop of edges given', type=float, required=False)

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
        # gpus=1,
        callbacks=[],
        logger=False,
        )
    prop = args.prop or conf['test_data']['prop_max_edges_given']
    data = SketchGraphDataModule(conf,preprocessing_params)
    _ = trainer.test(sketchPredictionmodel, dataloaders=data.test_dataloader())
    
    results = sketchPredictionmodel.test_results
    avg_precision, avg_recall = sketch_wise_precision_recall(results, thr=args.thr)
    print(f'Prop hidden: {prop:0.2}, avg precision = {avg_precision:0.3}, avg_recall = {avg_recall:0.3}')