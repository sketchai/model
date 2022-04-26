import logging 
import pytorch_lightning as pl
import torch
import os 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


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
# model.to(device)

from src.models.predict import PredictSketch
conf['edge_idx_map'] = preprocessing_params.get('edge_idx_map')
sketchPredictionmodel = PredictSketch(model, conf)
logger.info('-- Model initialization: end')

# Initialize a trainer
logger.info('-- Logger and Trainer initialization:...')
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint

logger_conf = conf.get('logger')
logger_tensorboard = TensorBoardLogger(save_dir = logger_conf.get('save_dir'), name = logger_conf.get('name'), log_graph=False)
profiler = PyTorchProfiler(profile_memory=True,export_to_chrome=True,schedule=torch.profiler.schedule(wait=1, warmup=1, active=5))

checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                        dirpath=logger_conf.get('save_dir'),
                                        filename='gat-{epoch:02d}-{val_loss:.2f}',
                                        mode="max",
                                        save_weights_only=True)
if __name__=='__main__':
    trainer = pl.Trainer(
        # accelerator='gpu',
        # devices=4,
        # strategy='dp',
        gpus=1,
        max_epochs=20, 
        # progress_bar_refresh_rate=20, 
        logger=logger_tensorboard,
        limit_train_batches=200,
        limit_val_batches=10,
        # callbacks=[checkpoint_callback],
        profiler=profiler,
    )
    logger.info('-- Logger and Trainer initialization: end')

    # Train the model 
    logger.info('-- Model fit: ...')
    trainer.fit(sketchPredictionmodel, datamodule=data)
    logger.info('-- Model fit: Done')
    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(os.path.join(logger_conf.get('save_dir'),'model_scripted.pt')) # Save
    # torch.save(model.state_dict(), os.path.join(logger_conf.get('save_dir'),'model_scripted.pt'))

    ######## STEP 3 : Compute validation
    # trainer.test(test_dataloaders=test)