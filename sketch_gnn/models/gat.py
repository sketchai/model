import torch
from typing import Dict
import pytorch_lightning as pl
import torch_geometric as pyg

from sketch_gnn.models.gat_versions.og_gat import GaT as og_GaT
from sketch_gnn.models.gat_versions.new_gatv1 import GaT as GaTv1
from sketch_gnn.models.gat_versions.new_gatv2 import GaT as GaTv2

import logging
logger = logging.getLogger(__name__)

class GaT(pl.LightningModule):
    """
    Neural net versioning helper class.

    This is mainly for prototyping purposes.
    Once a final GaT version have been chosen one can overwrite this file with the corresponding gat.py
    """

    def __init__(self, d_model: Dict = {}, d_prep: Dict = {}):
        """
        """
        super().__init__()
        version = d_model.get('version')
        if version == 'Og bis':
            og_GaT.__init__(self, d_model, d_prep)
            self.forward = lambda data: og_GaT.forward(self,data)
            self.embeddings = lambda data: og_GaT.embeddings(self,data)
        elif version == 'v2':
            GaTv2.__init__(self, d_model, d_prep)
            self.forward = lambda data: GaTv2.forward(self,data)
            self.embeddings = lambda data: GaTv2.embeddings(self,data)
        elif 'ConcatLinear' in version:
            GaTv1.__init__(self, d_model, d_prep)
            self.forward = lambda data: GaTv1.forward(self,data)
            self.embeddings = lambda data: GaTv1.embeddings(self,data)
        else:
            raise Exception(f'Couldn\'t find a version that matches {version}')

    def forward(self, data) -> Dict:
        raise NotImplementedError

    @torch.no_grad()
    def embeddings(self, data):
        raise NotImplementedError

    def representation_final_edges(x, edges_neg, edges_pos):
        return {
            'edges_neg': torch.index_select(x, 0, edges_neg[:, 0]) * torch.index_select(x, 0, edges_neg[:, 1]),
            'edges_pos': torch.index_select(x, 0, edges_pos[:, 0]) * torch.index_select(x, 0, edges_pos[:, 1])
            }


    def loss(prediction, data, coef_neg=1., weight_types=None):
        # device = data.edges_toInf_pos_types.device
        loss_edge_pos = torch.mean(torch.nn.functional.softplus(-prediction['edges_pos']))
        loss_edge_neg = torch.mean(torch.nn.functional.softplus(prediction['edges_neg']))

        loss_type = torch.nn.functional.cross_entropy(prediction['type'], data.constr_toInf_pos_types, weight=weight_types)
        
        return loss_edge_pos, coef_neg * loss_edge_neg, loss_type
