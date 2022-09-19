from collections import defaultdict
from typing import Dict
import numpy as np
import pytorch_lightning as pl
import logging
import tensorboard
import torch
import matplotlib.pyplot as plt
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

except ModuleNotFoundError:
    confusion_matrix = None

from sketch_gnn.inference.metrics import sketch_wise_precision_recall
from sketch_gnn.models.gat import GaT
from sketch_gnn.utils.to_dict import stack_hparams
logger = logging.getLogger(__name__)


class PredictSketch(pl.LightningModule):
    def __init__(self,model: object, conf: Dict = None):
        super().__init__()
        self.model = model
        # If conf is None, only the test loop will be available
        if conf:
            self.d_optimizer = conf.get('optimizer')
            self.coef_neg = self.d_optimizer.get('coef_neg')
            self.edge_idx_map = conf.get('edge_idx_map')
            self.node_idx_map = conf.get('node_idx_map')
            self.max_epochs = conf['train']['max_epochs']
            self.save_hyperparameters(stack_hparams(conf))
    

    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.d_optimizer.get('lr'))
        optimizers = [adam_optimizer]
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, factor=0.5, patience=self.d_optimizer.get('scheduler_step')), 
                         "monitor": "val/loss"}
        return optimizers, lr_schedulers


    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/val_loss": 1.})


    def training_step(self, batch, batch_idx):
        prediction = self.model(batch)
        loss = GaT.loss(prediction, batch, coef_neg=self.coef_neg, weight_types=None)
        e_pos_loss, e_neg_loss, e_type_loss = loss
        self.log('train/loss', sum(loss), batch_size = batch.num_graphs)
        self.log('train/edge_pos_loss', e_pos_loss, batch_size = batch.num_graphs)
        self.log('train/edge_neg_loss', e_neg_loss, batch_size = batch.num_graphs)
        self.log('train/edge_type_loss', e_type_loss, batch_size = batch.num_graphs)

        if batch_idx%100 == 0:
            edges_pos = prediction['edges_pos'].cpu().detach().numpy()
            edges_neg = prediction['edges_neg'].cpu().detach().numpy()
            self.log_binary_classification(edges_pos, edges_neg, tag='train', batch_size=batch.num_graphs)
            #TODO add aggregation (for now only one batch is used to compute the curve)
            frequency = self.max_epochs//10 or 1
            if batch_idx==0 and self.current_epoch%frequency==0:
                self.log_pr_curve(edges_pos, edges_neg, tag='train')
        return sum(loss)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.model(batch)
            loss = GaT.loss(output, batch, coef_neg=self.coef_neg, weight_types=None)
            edges_pos = output['edges_pos'].cpu().detach().numpy()
            edges_neg = output['edges_neg'].cpu().detach().numpy()
            predicted_type_pos = torch.argmax(output['type'], dim=-1).cpu().detach().numpy()
            predicted_type_neg = torch.argmax(output['type_neg'], dim=-1).cpu().detach().numpy()
            true_type = batch.constr_toInf_pos_types.cpu().detach().numpy()

        self.log('val/loss', sum(loss), batch_size=batch.num_graphs)
        self.log('hp/val_loss', sum(loss), batch_size=batch.num_graphs)

        self.log_binary_classification(edges_pos, edges_neg, tag='val', batch_size=batch.num_graphs)
        self.log_multiclass(true_type, predicted_type_pos, tag='val', batch_size=batch.num_graphs)

        if self.current_epoch == self.max_epochs-1 and batch_idx==0:
            self.log_embeddings(batch)

        #TODO add aggregation (for now only one batch is used to compute the matrix)
        #TODO add aggregation (for now only one batch is used to compute the curve)
        frequency = self.max_epochs//10 or 1
        if batch_idx==0 and self.current_epoch%frequency==0:
            self.log_pr_curve(edges_pos, edges_neg, tag='val')
            self.log_confusion_matrix(true_type, predicted_type_pos, tag='val')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        output = self.model(batch)
        output['constr_toInf_pos_types'] = batch.constr_toInf_pos_types
        output['n_edges_pos'] = torch.bincount(batch.constr_toInf_pos_batch)
        output['n_edges_neg'] = torch.bincount(batch.constr_toInf_neg_batch)
        output = {k:v.cpu() for k,v in output.items()}
        return output
    
    def test_epoch_end(self, outputs):
        cat_outputs = defaultdict(list)
        for output in outputs:
            for key, value in output.items():
                cat_outputs[key].append(value)
        cat_outputs = {k:torch.cat(v).numpy() for k,v in cat_outputs.items()}
        self.test_results = cat_outputs


    def log_binary_classification(self, edges_pos, edges_neg, tag, batch_size, on_step=False, on_epoch=True, thr=0.95):
        sigmoid = lambda x: 1/(1+np.exp(-x))
        tp = np.sum(sigmoid(edges_pos) > thr)
        fn = np.sum(sigmoid(edges_pos) < thr)
        tn = np.sum(sigmoid(edges_neg) < thr)
        fp = np.sum(sigmoid(edges_neg) > thr)

        if tp>0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + fn + tn + fp)
            self.log(f'{tag}/bin_accuracy', accuracy, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)
            self.log(f'{tag}/bin_precision', precision, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)
            self.log(f'{tag}/bin_recall',recall, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)

    def log_pr_curve(self, edges_pos, edges_neg, tag):
        tb_logger = self.logger.experiment
        label = np.concatenate([np.ones([len(edges_pos)]), np.zeros([len(edges_neg)])], axis = 0)
        scalar_pred = np.concatenate([edges_pos,edges_neg], axis = 0).flatten()
        predictions = 1 / (1 + np.exp(-scalar_pred))
        tb_logger.add_pr_curve(f'{tag}/pr_curve', label, predictions,global_step=self.global_step)

    def log_multiclass(self,true_type, predicted_type, tag, batch_size):
        accuracy = np.mean(predicted_type==true_type)
        self.log(f'{tag}/class_accuracy', accuracy, batch_size=batch_size)

    def log_confusion_matrix(self,true_type, predicted_type, tag):
        tb_logger = self.logger.experiment
        label_names = list(self.edge_idx_map.keys())[:-1]
        if confusion_matrix is not None:
            cm = confusion_matrix(true_type, predicted_type, labels=np.arange(len(label_names)), normalize='pred')
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=label_names)
            fig, ax = plt.subplots()
            disp.plot(ax=ax,xticks_rotation='vertical',include_values=False,)
            tb_logger.add_figure(f'{tag}/confusion_matrix',fig,global_step=self.global_step)

    def log_embeddings(self, data, max_size=1000):
        node_features = data.x[:,0].detach().cpu().numpy()
        embeddings = self.model.embeddings(data)

        # limit number of data points to display
        if max_size is not None:
            for key, value in embeddings.items():
                embeddings[key] = value[:max_size]
            node_features = node_features[:max_size]

        tb_logger = self.logger.experiment
        
        node_embeddings = {k:v for k,v in embeddings.items() if 'node' in k}
        node_inverse_map = {i: t for t, i in self.node_idx_map.items()}
        node_label_names = [node_inverse_map[k] for k in node_features]
        for key, array in node_embeddings.items():
            tb_logger.add_embedding(array,metadata=node_label_names,
                global_step=self.global_step,
                tag=f'epoch_{self.current_epoch:03d}_{key}')

        array = self.model.edge_embedding_layer.weight
        edge_label_names = [name for name in self.edge_idx_map]
        tb_logger.add_embedding(array,metadata=edge_label_names,
            global_step=self.global_step,
            tag=f'epoch_{self.current_epoch:03d}_edge_embeddings')

        # pos_embedd = embeddings['edges_pos_after_transformer']
        # neg_embedd = embeddings['edges_neg_after_transformer']
        # array = np.concatenate([pos_embedd, neg_embedd], axis=0)
        # edge_label = ['pos']*pos_embedd.shape[0] + ['neg']*neg_embedd.shape[0]
        # tb_logger.add_embedding(array,metadata=edge_label,
        #     global_step=self.global_step,
        #     tag=f'epoch_{self.current_epoch:03d}_binary_edge_classification')

        
        key = 'edges_pos_before_classif'
        array = embeddings[key]
        edge_inverse_map = {i: t for t, i in self.edge_idx_map.items()}
        print(embeddings['inferred_edges_pos_type'].shape)
        edge_label_names = [edge_inverse_map[k] for k in embeddings['inferred_edges_pos_type']]
        tb_logger.add_embedding(array,metadata=edge_label_names,
            global_step=self.global_step,
            tag=f'epoch_{self.current_epoch:03d}_edge_type_inference')
