import numpy as np
import torch
from sketch_gnn.models.gat import AttrDict
import logging
logger = logging.getLogger(__name__)

class EvalPrediction:

    def __init__(self, batch, output, edge_idx_map, threshold=0.95):
        batch = AttrDict(batch)

        self.edge_idx_map = edge_idx_map
        reverse_edge_idx_map = {i:k for k,i in edge_idx_map.items()}
        edges_pos = output['edges_pos'].cpu().detach().numpy()
        edges_neg = output['edges_neg'].cpu().detach().numpy()
        predicted_type_pos = torch.argmax(output['type'], dim=-1).cpu().detach().numpy()
        predicted_type_neg = torch.argmax(output['type_neg'], dim=-1).cpu().detach().numpy()

        predicted_linear_label = np.concatenate([edges_pos, edges_neg], axis=0).flatten()
        self.predicted_sigmoid = 1 / (1 + np.exp(-predicted_linear_label))
        self.true_label = np.concatenate([np.ones([len(edges_pos)]), np.zeros([len(edges_neg)])], axis=0)
        self.predicted_type = np.concatenate([predicted_type_pos, predicted_type_neg], axis=0)
        self.predicted_type_name = [reverse_edge_idx_map[i] for i in self.predicted_type]

        n_edges_given = batch.incidences.shape[1]//2
        edges_given = torch.transpose(batch.incidences,0,1)[:n_edges_given]
        self.references = torch.cat([batch.edges_toInf_pos, batch.edges_toInf_neg, edges_given], axis=0).cpu().detach().numpy()

        padding = 9999*torch.ones_like(batch.edges_toInf_neg, dtype=torch.int64)[:,0].flatten()
        self.true_type = torch.cat([batch.edges_toInf_pos_types, padding, batch.edge_features[:n_edges_given]], axis=0).cpu().detach().numpy()
        self.true_type_name = [reverse_edge_idx_map.get(i,'None') for i in self.true_type]

        if threshold is not None:
            self.set_categories(threshold)
            self.edges_category = np.empty_like(self.true_type, dtype=object)
            for key, l_idxes in self.d_categories.items():
                self.edges_category[l_idxes] = key

    def set_categories(self, threshold)->None:
        """
        Sort edges between categories

        #TODO add true negatives
        """
        self.edges_inferred = np.where(self.predicted_sigmoid > threshold)[0]
        n_edges_pos = int(sum(self.true_label))
        n_edges_neg = int(sum(self.true_label==0))
        self.edges_missed = np.array([i for i in range(n_edges_pos) if i not in self.edges_inferred])

        edges_idx = {
            'true_positives':               [],
            'false_positives':              [],
            'true_positives_wrong_type':    [],
            'false_negatives':              [],
            'false_negatives_wrong_type':   [],
            'given':                        [],
            'true_negatives':               [],
        }
        for i in self.edges_inferred:
            if self.true_label[i] == 1:
                if self.true_type[i] == self.predicted_type[i]:
                    edges_idx['true_positives'].append(i)
                else:
                    edges_idx['true_positives_wrong_type'].append(i)
            else:
                edges_idx['false_positives'].append(i)

        for i in self.edges_missed:
            if self.true_type[i] == self.predicted_type[i]:
                edges_idx['false_negatives'].append(i)
            else:
                edges_idx['false_negatives_wrong_type'].append(i)
        
        idx_min = self.true_label.shape[0]
        idx_max = self.true_type.shape[0]
        edges_idx['given'] = list(range(idx_min, idx_max))

        edges_idx['true_negatives'] = [i for i in range(n_edges_pos,n_edges_pos+n_edges_neg) 
            if i not in self.edges_inferred]

        self.d_categories = edges_idx

    def __getitem__(self,idx):
        """
        Returns a dict with information for the ith edge

        index is the concatenation of:    edges_toInf_pos | edges_toInf_neg | edges_given
        """
        category = self.edges_category[idx]

        info = {}
        attributes = ['predicted_sigmoid','true_label','predicted_type_name','true_type_name','references']
        map_info = np.array([
        #    pos   neg   given
            [True, True, False], # predicted_sigmoid
            [True, True, False], # true_label
            [True, True, False], # predicted_type_name
            [True, False, True], # true_type_name
            [True, True, True],  # references
            ])
        if category in ['true_positives','true_positives_wrong_type','false_negatives','false_negatives_wrong_type']:
            attributes_idx = np.nonzero(map_info[:,0])[0]
        elif category in ['false_positives', 'true_negatives']:
            attributes_idx = np.nonzero(map_info[:,1])[0]
        elif category in ['given']:
            attributes_idx = np.nonzero(map_info[:,2])[0]

        info['category'] = category
        for att_idx in attributes_idx:
            key = attributes[att_idx]
            info[key] = self.__dict__[key][idx]
        return info

    def __len__(self):
        return self.references.shape[0]

    def print_prediction(self):
        print('-' * 10 + 'INFERRED' + '-' * 10)
        for i in sorted(self.edges_inferred, key=lambda i: self.predicted_sigmoid[i], reverse=True):
            predicted_bin_score = self.predicted_sigmoid[i]
            predicted_type_name = self.predicted_type_name[i]
            true_type_name = self.true_type_name[i]
            print(f'Label: {true_type_name:<30}inferred: {predicted_type_name:<30}\
                score= {predicted_bin_score:<10.2f}{"INCORRECT" if true_type_name!=predicted_type_name else ""}')

        print('\n')
        print('-' * 10 + 'MISSED' + '-' * 10)
        for i in sorted(self.edges_missed, key=lambda i: self.predicted_sigmoid[i], reverse=True):
            predicted_bin_score = self.predicted_sigmoid[i]
            predicted_type_name = self.predicted_type_name[i]
            true_type_name = self.true_type_name[i]
            print(f'Label: {true_type_name:<30}inferred: {predicted_type_name:<30}\
                score= {predicted_bin_score:<10.2f}{"INCORRECT" if true_type_name!=predicted_type_name else ""}')

    def get_d_info(self):
        sorted_edges_info = {}
        for category, l_idxes in self.d_categories.items():
            array_of_idxes = np.array(l_idxes)
            sorted_edges_info[category] = self[array_of_idxes]
        return sorted_edges_info

