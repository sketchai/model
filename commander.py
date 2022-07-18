import logging 
import pytorch_lightning as pl
import torch
import os 
import pickle
import sketch_gnn
from sketch_gnn.utils.to_dict import parse_config
from sketch_gnn.dataloader.generate_dataModule import SketchGraphDataModule
from sketch_gnn.models.gat import GaT
from sketch_gnn.models.predict import PredictSketch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.profiler import schedule

logging.basicConfig(level=logging.WARNING)
logging.getLogger(sketch_gnn.__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

######## STEP 1 : Init Datasets
conf = parse_config('config/gat.yml')

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
sketchPredictionmodel = PredictSketch(model, conf)

logger.info('-- Model initialization: end')


######## STEP 3 : Init Trainer
logger.info('-- Logger and Trainer initialization:...')
logger_conf = conf.get('logger')
logger_tensorboard = TensorBoardLogger(
    save_dir = logger_conf.get('save_dir'),
    name = logger_conf.get('name'),
    log_graph=False,
    default_hp_metric=False)
# scheduler = schedule(wait=1, warmup=2, active=8)
# profiler = PyTorchProfiler(profile_memory=True,export_to_chrome=True,schedule=scheduler)

checkpoint_callback = ModelCheckpoint(monitor='val/loss',
                                        filename='gat-{epoch:02d}-{val_loss:.2f}',
                                        mode="min",
                                        save_weights_only=True)
if __name__=='__main__':
    trainer = pl.Trainer(
        # accelerator='gpu',
        # devices=4,
        # strategy='ddp_spawn',
        gpus=1,
        max_epochs=d_train.get('max_epochs'), 
        # progress_bar_refresh_rate=20, 
        logger=logger_tensorboard,
        # limit_train_batches=5,
        # limit_val_batches=5,
        callbacks=[checkpoint_callback],
        profiler='simple',
    )
    logger.info('-- Logger and Trainer initialization: end')

    # Train the model 
    logger.info('-- Model fit: ...')
    trainer.fit(sketchPredictionmodel, datamodule=data)
    logger.info('-- Model fit: Done')
    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(os.path.join(logger_conf.get('save_dir'),'model_scripted.pt')) # Save
    # torch.save(model.state_dict(), os.path.join(logger_conf.get('save_dir'),'model_scripted.pt'))