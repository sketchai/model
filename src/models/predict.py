import numpy as np
import torch
from typing import Dict
import pytorch_lightning as pl
import logging

from src.models.gat import GaT

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

class PredictSketch(pl.LightningModule):
    def __init__(self,model: object, conf: Dict = None):
        super().__init__()
        self.model = model
        self.d_optimizer = conf.get('optimizer')

        d_validation = conf.get('val_data')
        self.coef_neg = d_validation.get('coef_neg')

        


    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.d_optimizer.get('lr'))
        optimizers = [adam_optimizer]
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, factor=0.5, patience=self.d_optimizer.get('scheduler_step')), 
                         "monitor": "metric_to_track"}
        return optimizers, lr_schedulers


    def training_step(self, batch, batch_idx):
        prediction = self.model(batch)
        # logger.debug(f'Prediction: {prediction}')
        loss = GaT.loss(prediction, batch, coef_neg=self.coef_neg, weight_types=None)
        # Save loss
        self.logger.log_metrics({'train_loss': loss})
        return loss 


    def validation_step(self, batch, batch_idx):
        # result = pl.EvalResult()


        with torch.no_grad():
            prediction = self.model(batch)
            loss = GaT.loss(prediction, batch, coef_neg=self.coef_neg, weight_types=None).item()
            perfs = GaT.performances(prediction, batch)

            # Save loss
            self.logger.agg_and_log_metrics({'val_loss': loss,
                                     'val_perf_edge.n_edges_pos_predicted_pos': perfs[0][0],
                                     'val_perf_edge.n_edges_predicted_pos': perfs[0][1],
                                     'val_perf_edge.n_edges_pos': perfs[0][2]})

            x = perfs[1]
            self.log('metric_to_track',loss)
            # logger.debug(f'val_perf_type {x}')



