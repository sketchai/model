from tying import Dict 
import numpy as np
import torch
from typing import Dict
import pytorch_lightning as pl

from src.models.numerical_features.generator import generate_embedding
from src.models.dense_emb import DenseSparsePreEmbedding, ConcatenateLinear


class PredictSketch(pl.LightningModule):
    def __init__(self,model: object, conf: Dict = None):
        self.model = model
        self.d_optimizer = conf.get('optimizer')

    def configure_optimizers(self):
    optimizer = torch.optim.Adam(model.parameters(), lr=self.d_optimizer['lr'])
    if self.d_optimizer['scheduler_lr']:
        scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=self.d_optimizer['scheduler_step'])
        return [optimizer], [scheduler_lr]

    def training_step(self, batch, batch_idx):
        prediction = self.model(batch)
        loss = GaT.loss(prediction, batch, coef_neg=coef_neg, weight_types=weight_types)
        return loss 


    def validation_step(self, batch, batch_idx):
        result = pl.EvalResult()

        with torch.no_grad():
            prediction = self.model(batch)
            loss = GravTransformer.loss(prediction, batch, coef_neg=coef_neg, weight_types=weight_types).item()
            perfs = GravTransformer.performances(prediction, batch)

            # Save loss
            result.log('val_loss', val_loss)
            result.log('val_perf_edge', perfs[0])
            result.log('val_perf_type', perfs[1])



