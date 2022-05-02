from typing import Dict
import numpy as np
import pytorch_lightning as pl
import logging
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

        self.log_binary_classification(edges_pos, edges_neg, tag='val')

        self.log_multiclass(true_type, predicted_type, tag='val')

    def log_binary_classification(self, edges_pos, edges_neg, tag):
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
        label = np.concatenate([np.ones([len(edges_pos)]), np.zeros([len(edges_neg)])], axis = 0)
        scalar_pred = np.concatenate([edges_pos,edges_neg], axis = 0).flatten()
        predictions = 1 / (1 + np.exp(-scalar_pred))
        tensorboard.add_pr_curve(f'{tag}/pr_curve', label, predictions,global_step=self.global_step)

    def log_multiclass(self,true_type, predicted_type, tag):
        tensorboard = self.logger.experiment
        label_names = list(self.edge_idx_map.keys())[:-1]

        accuracy = np.mean(predicted_type==true_type)
        self.log(f'{tag}/class_accuracy', accuracy)
        if confusion_matrix is not None:
            cm = confusion_matrix(true_type, predicted_type, labels=np.arange(len(label_names)), normalize='pred')
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=label_names)
            fig, ax = plt.subplots()
            disp.plot(ax=ax,xticks_rotation='vertical',include_values=False,)
            tensorboard.add_figure('confusion_matrix',fig,global_step=self.global_step)
