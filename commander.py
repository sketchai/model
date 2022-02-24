import logging 
import pytorch_lightning as pl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

#### SG path
import sys 
sys.path.append('/home/i37181/Documents/Projets/CAO/SketchGraphs/sketchgraphs')

######## STEP 1 : Import Datasets
from src.utils.to_dict import yaml_to_dict
conf = yaml_to_dict('config/gat.yml')


# Initialiaze parameters
d_train = conf.get('train')
import pickle
with open(d_train.get('prep_parms_path'), 'rb') as f:
    preprocessing_params = pickle.load(f)

lMax = preprocessing_params.get('lMax')
node_feature_mapping_dim = preprocessing_params.get('node_feature_dimensions')
edge_feature_mapping_dim = preprocessing_params.get('edge_feature_dimensions')

from src.dataloader.collate import collate 
# collate all examples in one batch
import functools
collate_fn = functools.partial(collate, node_feature_dims=node_feature_mapping_dim,
                                   edge_feature_dims=edge_feature_mapping_dim, lMax=lMax,
                                   prop_max_edges_given=d_train.get('prop_max_edges_given'))

from src.dataloader.load import generate_dataset

logger.info('-- Load Train Set')
train = generate_dataset(conf=conf.get("train_data"), batch_size=d_train.get('batch_size'), collate_fn=collate_fn)

logger.info('-- Load Validation Set')
val = generate_dataset(conf=conf.get("val_data"), batch_size=d_train.get('batch_size'), collate_fn=collate_fn)


######## STEP 2 : Initialize a trained
from src.models.transf_graph import GaT
gat = GaT()
# Model initialization
model = GravTransformer(node_feature_mapping_dim,
                            edge_feature_mapping_dim,
                            arch['embedding_dim'],
                            arch['n_head'],
                            arch['num_layers'],
                            arch['positional_encoding'],
                            lMax)
model.to(device)

sketchPredictionmodel = PredictSketch(gat_model)

# Initialize a trainer
from pytorch_lightning.loggers import TensorBoardLogger

logger_conf = cong.get('logger')
logger = TensorBoardLogger(save_dir = logger_conf.get('save_dir'), name = logger_conf.get('name'))
trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20, automatic_optimization=False, logger=logger)

# Train the model 
trainer.fit(sketchPredictionmodel, train, val)


######## STEP 3 : Compute validation
trainer.test(test_dataloaders=test)