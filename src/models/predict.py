import numpy as np
import torch
from typing import Dict
import pytorch_lightning as pl
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

class PredictSketch(pl.LightningModule):
    def __init__(self,model: object, conf: Dict = None):
        super().__init__()
        self.model = model
        self.d_optimizer = conf.get('optimizer')


    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.d_optimizer.get('lr'))
        optimizers = [adam_optimizer]
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, factor=0.5, patience=self.d_optimizer.get('scheduler_step')), 
                        "monitor": "metric_to_track"}
        return optimizers, lr_schedulers


    def training_step(self, batch, batch_idx):
        prediction = self.model(batch)
        loss = GaT.loss(prediction, batch, coef_neg=coef_neg, weight_types=weight_types)
        return loss 

    def validation_step(self, batch, batch_idx):
        # result = pl.EvalResult()


        with torch.no_grad():
            prediction = self.model(batch)
            loss = GravTransformer.loss(prediction, batch, coef_neg=coef_neg, weight_types=weight_types).item()
            perfs = GravTransformer.performances(prediction, batch)

            # Save loss
            self.log('val_loss', val_loss)
            self.log('val_perf_edge', perfs[0])
            self.log('val_perf_type', perfs[1])



