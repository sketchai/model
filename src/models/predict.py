from typing import Dict
import numpy as np
import pytorch_lightning as pl
import logging
import tensorboard
import torch
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

except ModuleNotFoundError:
    confusion_matrix = None

from src.models.gat import GaT

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class PredictSketch(pl.LightningModule):
    def __init__(self,model: object, conf: Dict = None):
        super().__init__()
        self.model = model
        self.d_optimizer = conf.get('optimizer')

        d_validation = conf.get('val_data')
        self.coef_neg = d_validation.get('coef_neg')
        self.edge_idx_map = conf.get('edge_idx_map')
        self.node_idx_map = conf.get('node_idx_map')
    

    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.d_optimizer.get('lr'))
        optimizers = [adam_optimizer]
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, factor=0.5, patience=self.d_optimizer.get('scheduler_step')), 
                         "monitor": "metric_to_track"}
        return optimizers, lr_schedulers


    def training_step(self, batch, batch_idx):
        prediction = self.model(batch)

        loss = GaT.loss(prediction, batch, coef_neg=self.coef_neg, weight_types=None)
        # Save loss
        self.log('train_loss', loss, batch_size = batch['l_batch'])
        return loss 


    def validation_step(self, batch, batch_idx):
        # result = pl.EvalResult()

        with torch.no_grad():
            prediction = self.model(batch)
            l_batch = batch['l_batch']
            loss = GaT.loss(prediction, batch, coef_neg=self.coef_neg, weight_types=None).item()
            perfs = GaT.performances(prediction, batch, self.edge_idx_map)
        edges_pos, edges_neg, predicted_type, true_type = perfs
        self.log('metric_to_track',loss)
        self.log('val_loss', loss)

        self.log_binary_classification(edges_pos, edges_neg, tag='val',batch_idx=batch_idx)

        self.log_multiclass(true_type, predicted_type, tag='val',batch_idx=batch_idx)

        if (self.current_epoch+1)%5==0 and batch_idx==0:
            self.log_embeddings(batch, tag='val')

    def log_binary_classification(self, edges_pos, edges_neg, tag, batch_idx):
        tensorboard = self.logger.experiment
        tp = np.sum(edges_pos > 0)
        fn = np.sum(edges_pos < 0)
        tn = np.sum(edges_neg < 0)
        fp = np.sum(edges_neg > 0)

        if tp>0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + fn + tn + fp)
            self.log(f'{tag}/bin_accuracy', accuracy)
            self.log(f'{tag}/bin_precision', precision)
            self.log(f'{tag}/bin_recall',recall)

        #TODO add aggregation (for now only one batch is used to compute the curve)
        if batch_idx==0:
            label = np.concatenate([np.ones([len(edges_pos)]), np.zeros([len(edges_neg)])], axis = 0)
            scalar_pred = np.concatenate([edges_pos,edges_neg], axis = 0).flatten()
            predictions = 1 / (1 + np.exp(-scalar_pred))
            tensorboard.add_pr_curve(f'{tag}/pr_curve', label, predictions,global_step=self.global_step)

    def log_multiclass(self,true_type, predicted_type, tag, batch_idx):
        tensorboard = self.logger.experiment
        label_names = list(self.edge_idx_map.keys())[:-1]

        accuracy = np.mean(predicted_type==true_type)
        self.log(f'{tag}/class_accuracy', accuracy)
        #TODO add aggregation (for now only one batch is used to compute the matrix)
        if confusion_matrix is not None and batch_idx==0:
            cm = confusion_matrix(true_type, predicted_type, labels=np.arange(len(label_names)), normalize='pred')
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=label_names)
            fig, ax = plt.subplots()
            disp.plot(ax=ax,xticks_rotation='vertical',include_values=False,)
            tensorboard.add_figure('confusion_matrix',fig,global_step=self.global_step)

    def log_embeddings(self, batch,tag='',max_size=1000):
        node_features = batch['node_features'].detach().cpu().numpy()
        edge_features = batch['edge_features'].detach().cpu().numpy()
        embeddings = self.model.embeddings(batch)

        # limit number of data points to display
        if max_size is not None:
            for key, value in embeddings.items():
                embeddings[key] = value[:max_size]
            node_features = node_features[:max_size]
            edge_features = edge_features[:max_size]

        tensorboard = self.logger.experiment
        
        node_embeddings = {k:v for k,v in embeddings.items() if 'node' in k}
        node_inverse_map = {i: t for t, i in self.node_idx_map.items()}
        node_label_names = [node_inverse_map[k] for k in node_features]
        for key, array in node_embeddings.items():
            tensorboard.add_embedding(array,metadata=node_label_names,
                global_step=self.global_step,
                tag=f'epoch_{self.current_epoch:03d}/{key}')

        key = 'edges_bf_msg_passing'
        array = embeddings[key]
        edge_inverse_map = {i: t for t, i in self.edge_idx_map.items()}
        edge_label_names = [edge_inverse_map[k] for k in edge_features]
        tensorboard.add_embedding(array,metadata=edge_label_names,
            global_step=self.global_step,
            tag=f'epoch_{self.current_epoch:03d}/{key}')

        pos_embedd = embeddings['edges_pos_after_transformer']
        neg_embedd = embeddings['edges_neg_after_transformer']
        array = np.concatenate([pos_embedd, neg_embedd], axis=0)
        edge_label = ['pos']*pos_embedd.shape[0] + ['neg']*neg_embedd.shape[0]
        tensorboard.add_embedding(array,metadata=edge_label,
            global_step=self.global_step,
            tag=f'epoch_{self.current_epoch:03d}/binary_edge_classification')

        
        key = 'edges_pos_after_transformer'
        array = embeddings[key]
        edge_inverse_map = {i: t for t, i in self.edge_idx_map.items()}
        print(embeddings['inferred_edges_pos_type'].shape)
        edge_label_names = [edge_inverse_map[k] for k in embeddings['inferred_edges_pos_type']]
        tensorboard.add_embedding(array,metadata=edge_label_names,
            global_step=self.global_step,
            tag=f'epoch_{self.current_epoch:03d}/edge_type_inference')