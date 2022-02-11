import numpy as np
#import torch
from scipy.optimize import linear_sum_assignment


class Meter(object):
    """Computes and stores the sum, average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_sum(self):
        return self.sum

    def value(self):
        """ Returns the value over one epoch """
        return self.avg

    def is_active(self):
        return self.count > 0


class ValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val


def make_meter_matching():
    meters_dict = {
        'loss': Meter(),
        'acc': Meter(),
        'perf_edge': Meter(),
        'perf_type': Meter()
        # 'acc_gr': Meter(),
        # 'batch_time': Meter(),
        # 'data_time': Meter(),
        # 'epoch_time': Meter(),
    }
    return meters_dict


def agreg_performances(perfs):
    """
    To aggregate the outputs of the function performances over many validation batches.
    """
    perfs = np.array(perfs)
    perfs = np.sum(perfs, axis=0)
    epsilon = 1e-5
    return perfs[0] / (perfs[1] + epsilon), perfs[0] / (perfs[2] + epsilon)
