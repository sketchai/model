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

# Example and create model
logger_conf = conf.get('logger')
filepath = os.path.join(logger_conf.get('save_dir'),'model.onnx')
train_data = data.train_dataloader()
for d in train_data:
    input_sample = d
    break
model = GaT(d_model, preprocessing_params, example=input_sample)
model.to(device)

from src.models.predict import PredictSketch
conf['edge_idx_map'] = preprocessing_params.get('edge_idx_map')
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
trainer = pl.Trainer(gpus=1, max_epochs=1, 
                    progress_bar_refresh_rate=20, 
                    logger=logger_tensorboard,
                    limit_train_batches=10,
                    limit_val_batches=10,
                    callbacks=[checkpoint_callback]
                    )

logger.info('-- Logger and Trainer initialization: end')

# Train the model 
logger.info('-- Model fit: ...')
trainer.fit(sketchPredictionmodel, datamodule=data)
logger.info('-- Model fit: Done')

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save(os.path.join(logger_conf.get('save_dir'),'model_scripted.pt')) # Save
torch.save(model.state_dict(), os.path.join(logger_conf.get('save_dir'),'model_scripted.pt'))



logger.info(f'sample data = {input_sample}')
input_nampes = []
dummy_inputs  = input_sample

# l_key = ['l_batch', 'node_features', 'sparse_node_features', 
#         'incidences', 'edge_features', 'sparse_edge_features', 
#         'edges_toInf_pos', 'edges_toInf_pos_types', 
#         'edges_toInf_neg', 'src_key_padding_mask', 'positions', 'is_given']

# test = {}
# for key in l_key[:1]:
#     test[key] = dummy_inputs.get(key)
#     if key == 'l_batch':
#         test[key] = [test[key]] #torch.tensor(test[key])

# here = torch._C._jit_flatten(test)
# logger.info(f'here = {here} ')

# raise Error('Stop code')

# model.to_onnx(filepath, dummy_inputs, export_params=True, verbose = True)
#, input_names = input_names, output_names = output_names)

script = model.to_torchscript()

# save for use in production environment
torch.jit.save(script, filepath)

######## STEP 3 : Compute validation
# trainer.test(test_dataloaders=test)