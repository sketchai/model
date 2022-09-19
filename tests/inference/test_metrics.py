import unittest
import logging
import torch
import numpy as np
from sketch_gnn.inference.metrics import sketch_wise_precision_recall
logger = logging.getLogger(__name__)

class TestMetrics(unittest.TestCase):
    def test_sk_wise_pr(self):
        outputs = {
            'edges_pos': [9,  8, -2,  2, 5],
            'edges_neg': [5, -4, -3, -2],
            'n_edges_pos': [3,2],
            'n_edges_neg': [2,2],
            'constr_toInf_pos_types': [1,2,3,4,5],
            'type': np.zeros((5,12)),            
        }
        indexes = [(0,1), (1,2), (2,3), (3,10), (4,5)]
        for idx in indexes:
            outputs['type'][idx] = 1
        outputs = {k:np.array(v) for k,v in outputs.items()}
        pre, rec = sketch_wise_precision_recall(outputs,thr=0.5)
        # sketch 1: [tp, tp, fn, fp, tn]
        # -> tp: 2, fp:1, fn: 1
        # -> tp: 2, predicted: 3, true_nb: 3

        # sketch 2: [fp&fn, tp, tn, tn]
        # sketch 2: tp: 1, fp:1, fn: 1
        # -> tp: 1, predicted: 2, true_nb: 2
        
        expected_pre = 1/2*(2/(2+1) + 1/(1+1))
        expected_rec = 1/2*(2/(2+1) + 1/(1+1))
        self.assertEqual(pre, expected_pre)
        self.assertEqual(rec, expected_rec)