import logging 
import pytorch_lightning as pl
import torch
import os 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

#### SG path
import sys
CAO_DIR = '/home/i37181/Documents/Projets/CAO/SketchGraphs'
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

# Add node_idx_map and edge_idx_map (must be placed directly into the preprocessing files)
from src.utils.maps import NODE_IDX_MAP, EDGE_IDX_MAP, PADDING_IDX
preprocessing_params['node_idx_map'] = NODE_IDX_MAP
preprocessing_params['edge_idx_map'] = EDGE_IDX_MAP
preprocessing_params['padding_idx'] = PADDING_IDX

logger.info(f'-- Load preprocessing params')
logger.info(f'-- -- list keys: {preprocessing_params.keys()}')
logger.info(f'-- -- preprocessing params: {preprocessing_params}')


# Create DataLoader
from src.dataloader.generate_dataModule import SketchGraphDataModule
data = SketchGraphDataModule(conf,preprocessing_params)


######## STEP 2 : Initialize a trained
from src.models.gat import GaT

logger.info('-- Model initialization:...')

# Model initialization
d_model = conf.get('model')

use_cuda = not d_model.get('cpu') and torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else 'cpu'

model = GaT(d_model, preprocessing_params)
model.to(device)

from src.models.predict import PredictSketch
sketchPredictionmodel = PredictSketch(model, conf)
logger.info('-- Model initialization: end')

# Initialize a trainer
logger.info('-- Logger and Trainer initialization:...')
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

logger_conf = conf.get('logger')
logger_tensorboard = TensorBoardLogger(save_dir = logger_conf.get('save_dir'), name = logger_conf.get('name'), log_graph=True)


checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                        dirpath=logger_conf.get('save_dir'),
                                        filename='gat-{epoch:02d}-{val_loss:.2f}',
                                        mode="max",
                                        save_weights_only=True)
trainer = pl.Trainer(gpus=1, max_epochs=2, 
                    progress_bar_refresh_rate=20, 
                    logger=logger_tensorboard,
                    limit_train_batches=100,
                    limit_val_batches=100,
                    callbacks=[checkpoint_callback]
                    )

logger.info('-- Logger and Trainer initialization: end')

# Train the model 
logger.info('-- Model fit: ...')
trainer.fit(sketchPredictionmodel, datamodule=data)
logger.info('-- Model fit: Done')

# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save(os.path.join(logger_conf.get('save_dir'),'model_scripted.pt')) # Save
torch.save(model.state_dict(), os.path.join(logger_conf.get('save_dir'),'model_scripted.pt'))

######## STEP 3 : Compute validation
# trainer.test(test_dataloaders=test)