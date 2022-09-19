from collections import defaultdict

import torch
import logging
import numpy as np
logger = logging.getLogger(__name__)


def sketch_wise_precision_recall(outputs, thr=0.95):
        """
        Same metric as in Sketchgraphs 2020, Seff et al.
        
        Our architecture requires a user-defined threshold.
        Another way would be to compute average over multiple thresholds.
        """
        cat_edges_pos = 1 / (1 + np.exp(-outputs['edges_pos']))
        cat_edges_neg = 1 / (1 + np.exp(-outputs['edges_neg']))
        n_edges_pos = outputs['n_edges_pos']
        n_edges_neg = outputs['n_edges_neg']
        cat_true_type = outputs['constr_toInf_pos_types']
        cat_predicted_type_pos = np.argmax(outputs['type'], axis=-1)
        logger.debug(f'types = {cat_predicted_type_pos}')
        precision=[]
        recall=[]
        n_pos_offset = 0
        n_neg_offset = 0
        for n_pos, n_neg in zip(n_edges_pos, n_edges_neg):
            edges_pos = cat_edges_pos[n_pos_offset:n_pos_offset+n_pos].reshape(-1)
            edges_neg = cat_edges_neg[n_neg_offset:n_neg_offset+n_neg].reshape(-1)
            true_type = cat_true_type[n_pos_offset:n_pos_offset+n_pos]
            predicted_type_pos = cat_predicted_type_pos[n_pos_offset:n_pos_offset+n_pos]
            n_pos_offset+=n_pos
            n_neg_offset+=n_neg
            tp = np.sum((edges_pos > thr) & (true_type == predicted_type_pos))
            fn = np.sum(edges_pos < thr) + np.sum((edges_pos > thr) & (true_type != predicted_type_pos))
            fp = np.sum(edges_neg > thr) + np.sum((edges_pos > thr) & (true_type != predicted_type_pos))
            logger.debug(f'tp: {tp}, fn: {fn}, fp: {fp}')
            recall.append(tp/ (tp + fn) if (tp+fn)>0 else 0)
            precision.append(tp/ (tp + fp) if (tp+fp)>0 else 0)

        return np.mean(precision), np.mean(recall)