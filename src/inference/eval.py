import numpy as np
import torch
from src.models.gat import AttrDict
from src.utils.logger import logger

class EvalPrediction:

    def __init__(self, batch, output, edge_idx_map, threshold=0.95):
        batch = AttrDict(batch)

        self.edge_idx_map = edge_idx_map
        edges_pos = output['edges_pos'].cpu().detach().numpy()
        edges_neg = output['edges_neg'].cpu().detach().numpy()
        predicted_type_pos = torch.argmax(output['type'], dim=-1).cpu().detach().numpy()
        predicted_type_neg = torch.argmax(output['type_neg'], dim=-1).cpu().detach().numpy()
        self.true_type = batch.edges_toInf_pos_types.cpu().detach().numpy()

        self.binary_pred = np.concatenate([edges_pos, edges_neg], axis=0).flatten()
        sigmoid_pred = 1 / (1 + np.exp(-self.binary_pred))
        self.true_label = np.concatenate([np.ones([len(edges_pos)]), np.zeros([len(edges_neg)])], axis=0)
        self.predicted_type = np.concatenate([predicted_type_pos, predicted_type_neg], axis=0)

        edges_inferred = np.where(sigmoid_pred > threshold)[0]
        self.edges_inferred = sorted(edges_inferred, key=lambda i: sigmoid_pred[i], reverse=True)

        edges_missed = np.array([i for i in range(len(edges_pos)) if i not in edges_inferred])
        self.edges_missed = sorted(edges_missed, key=lambda i: sigmoid_pred[i], reverse=True)

        self.all_references = torch.cat([batch.edges_toInf_pos, batch.edges_toInf_neg], axis=0)

    def sort_edges(self):
        """
        Sort edges between 5 categories, true negatives are not stored
        """

        sorted_edges_idx = {
            'true_positives':               [],
            'false_positives':              [],
            'true_positives_wrong_type':    [],
            'false_negatives':              [],
            'false_negatives_wrong_type':   [],
        }

        for i in self.edges_inferred:
            if self.true_label[i] == 1:
                if self.true_type[i] == self.predicted_type[i]:
                    sorted_edges_idx['true_positives'].append(i)
                else:
                    sorted_edges_idx['true_positives_wrong_type'].append(i)
            else:
                sorted_edges_idx['false_positives'].append(i)

        for i in self.edges_missed:
            if self.true_type[i] == self.predicted_type[i]:
                sorted_edges_idx['false_negatives'].append(i)
            else:
                sorted_edges_idx['false_negatives_wrong_type'].append(i)

        return sorted_edges_idx

    def print_prediction(self):
        edge_idx_map_reverse = list(self.edge_idx_map.keys())
        print('-' * 10 + 'INFERRED' + '-' * 10)
        for i in self.edges_inferred:
            predicted_bin_score = self.binary_pred[i]
            predicted_type_name = edge_idx_map_reverse[self.predicted_type[i]]
            if self.true_label[i] == 1:
                true_type_name = edge_idx_map_reverse[self.true_type[i]]
            else:
                true_type_name = 'None'
            print(f'Label: {true_type_name:<30}inferred: {predicted_type_name:<30}\
                score= {predicted_bin_score:<10.2f}{"INCORRECT" if true_type_name!=predicted_type_name else ""}')

        print('\n')
        print('-' * 10 + 'MISSED' + '-' * 10)
        for i in self.edges_missed:
            predicted_bin_score = self.binary_pred[i]
            predicted_type_name = edge_idx_map_reverse[self.predicted_type[i]]
            print(f'Label: {true_type_name:<30}inferred: {predicted_type_name:<30}\
                score= {predicted_bin_score:<10.2f}{"INCORRECT" if true_type_name!=predicted_type_name else ""}')

