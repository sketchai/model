import torch


def set_cuda_status(cpu: bool, logger: object = None):
    # check if cuda gpu are available
    if not cpu and torch.cuda.is_available()
    device = torch.device('cuda')
    else:
        device = 'cpu'
    if logger:
        logger.info('Using device:', device)
