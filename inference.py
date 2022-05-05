import logging 
import pytorch_lightning as pl
import torch
import os 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

FROM_CHECKPOINT = False


# Load the model
from src.models.gat import GaT

logger.info('-- Model initialization:...')

#### SG path
import sys
CAO_DIR = '/home/i37181/CAO_ML/git_depot/sketch_graph'
sys.path.append(os.path.join(CAO_DIR, 'sketchgraphs'))

######## STEP 1 : Import Datasets
from src.utils.to_dict import yaml_to_dict
conf = yaml_to_dict('config/gat.yml')

# update path
main_dir = conf.get('experiment').get('dir')
conf['logger']['save_dir'] = os.path.join(main_dir, conf['logger']['save_dir'])

conf['train']['prep_parms_path'] = os.path.join(main_dir, conf['train']['file_prep_parms'])
for x in ['data', 'weights']:
    conf['train_data'][f'path_{x}'] = os.path.join(main_dir, conf['train_data'][f'file_{x}'])    
    conf['val_data'][f'path_{x}'] = os.path.join(main_dir, conf['val_data'][f'file_{x}'])   


# Initialiaze parameters
d_train = conf.get('train')
import pickle
with open(d_train.get('prep_parms_path'), 'rb') as f:
    preprocessing_params = pickle.load(f)


logger.info(f'-- Load preprocessing params')
logger.info(f'-- -- list keys: {preprocessing_params.keys()}')
logger.info(f'-- -- preprocessing params: {preprocessing_params}')

# Model initialization
d_model = conf.get('model')

logger.info(f'd_model: {d_model}')
logger.info(f'preprocessing_params: {preprocessing_params}')
use_cuda = not d_model.get('cpu') and torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else 'cpu'

model = GaT(d_model, preprocessing_params)
model.to(device)

if FROM_CHECKPOINT:
    PATH= 'tests/asset/dataset/val/experiment/results/gat-epoch=00-val_loss=0.88.ckpt'
    model.load_from_checkpoint(PATH)
else :
    PATH= 'tests/asset/dataset/val/experiment/results/model_scripted.pt'
    model.load_state_dict(torch.load(PATH))

model.train(False)

# Infer
# conf = yaml_to_dict('config/gat.yml') #yaml_to_dict('tests/asset/mock/gat_example.yml')
# d_train = conf.get('train')
# with open(d_train.get('prep_parms_path'), 'rb') as f:
#     d_prep = pickle.load(f)

# logger.info(f'--- d_prep= {d_prep}')
from src.dataloader.generate_dataModule import SketchGraphDataModule

graph_dataset = SketchGraphDataModule(conf, preprocessing_params)
dataset = graph_dataset.val_dataloader()
logger.debug(f'dataset size={len(dataset)}')
logger.debug(f'dataset.dataset: nb size={len(dataset.dataset)}')

EDGE_IDX_MAP_REVERSE = {i: t for t, i in preprocessing_params.get('edge_idx_map').items()}

import numpy as np
import sys
CAO_DIR = '/home/i37181/Documents/Projets/CAO/SketchGraphs'
sys.path.append(os.path.join(CAO_DIR, 'sketchgraphs'))
from sketchgraphs.data import sequence

logger.info(f'here')
with torch.no_grad():
    for i, batch in enumerate(dataset):
        logger.info(f'batch: {batch.node_features[:10]}')
        logger.info(f'edges_toInf_pos: {batch.edges_toInf_pos}')
        logger.info(f'is_given: {batch.is_given}')
        prediction = model(batch)
    
        probs_edge = torch.sigmoid(prediction['edges_pos']).cpu().numpy()[:,0]
        probs_type = torch.nn.functional.softmax(prediction['type'], dim=-1).cpu().numpy()

        edges_pos = np.nonzero(probs_edge>0.5)[0]

        op_inferred = []
        for edge_pos in edges_pos:
            op_inferred.append(
                sequence.EdgeOp(EDGE_IDX_MAP_REVERSE[np.argmax(probs_type[edge_pos])],
                        references=tuple(batch.edges_toInf_pos[edge_pos].tolist()))
            )
        
        logger.info(f'prediction: {op_inferred}')
        if i > 3 :
            break
