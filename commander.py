from src.dataloader.graph_data import collate 

from sacred import SETTINGS
from sacred import Experiment
from loaders.graph_data import *
from models.transf_graph import *
from models.dense_emb import *
from toolbox import utils, logger, metrics
from sketchgraphs.data import flat_array, sequence
import os
import torch
import numpy as np
from functools import partial
import json
import functools
import pickle
import gzip
from src.utils.cuda_tools import set_cuda_status
logging.basicConfig(level=logging.DEBUG)
Logger = logging.getLogger()


SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# BEGIN Sacred setup
ex = Experiment()
ex.add_config('default.yaml')


@ex.config
def update_paths(root_dir, name):
    utils.check_dir(root_dir)
    log_dir = os.path.join(root_dir, name)


@ex.config_hook
def init_observers(config, command_name, logger):
    if command_name == 'training':
        neptune = config['observers']['neptune']
        if neptune['enable']:
            from neptunecontrib.monitoring.sacred import NeptuneObserver
            ex.observers.append(NeptuneObserver(project_name=neptune['project']))
    return config


@ex.post_run_hook
def clean_observer(observers):
    """ Observers that are added in a config_hook need to be cleaned """
    try:
        neptune = observers['neptune']
        if neptune['enable']:
            from neptunecontrib.monitoring.sacred import NeptuneObserver
            ex.observers = [obs for obs in ex.observers
                            if not isinstance(obs, NeptuneObserver)]
    except KeyError:
        pass


# END Sacred setup

# TO DO: ponderation dans la perte
coef_neg = 6
weight_types_d = {}  # sequence.ConstraintType.Vertical: 4,
#                sequence.ConstraintType.Perpendicular: 3,
#                sequence.ConstraintType.Length: 2.5,
#                sequence.ConstraintType.Diameter: 4}

# TO DO: mettre dans le folder approprie... et corriger! bug...


def schedule(epoch, num_epochs):
    """
    Apply a multiplicative coefficient to lr.
    """
    if epoch < 2 * num_epochs // 5:
        return 3**(epoch / (num_epochs // 5) - 2)
    elif epoch > 3 * num_epochs // 5:
        return 10**(3 - epoch / (num_epochs // 5))
    else:
        return 1


@ex.capture
def init_logger(name, _config, _run):
    # set loggers
    exp_logger = logger.Experiment(name, _config, run=_run)
    exp_logger.add_meters('train', metrics.make_meter_matching())
    #exp_logger.add_meters('val', metrics.make_meter_matching())
    exp_logger.add_meters('test', metrics.make_meter_matching())
    exp_logger.add_meters('hyperparams', {'learning_rate': metrics.ValueMeter()})
    return exp_logger


@ex.capture
def init_output_env(_config, log_dir):
    utils.check_dir(log_dir)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(_config, f)


@ex.command
def training(cpu: bool, arch, train, train_data, test_data, path_dataset):
    """
        cpu (boolean): indicates if cpu must be used or not
    """

    # Initialization
    init_output_env()  # create output logdir
    exp_logger = init_logger()  # init logger experiment + metrics

    # Initialiaze parameters
    with open(os.path.join(path_dataset, train_data.get('name_param')), 'rb') as f:
        preprocessing_params = pickle.load(f)
    lMax = preprocessing_params['lMax']
    node_feature_mapping_dim = preprocessing_params['node_feature_dimensions']
    edge_feature_mapping_dim = preprocessing_params['edge_feature_dimensions']

    # Dataset loading
    # TODO: Write a DataLoader
    # Train dataset
    ds_train = GraphDataset(os.path.join(path_dataset, train_data['name_data']),
                            os.path.join(path_dataset, train_data['name_weights']),
                            n_slice=train_data['n_slice'])
    # 'data/sg_t16_train/','data/sg_t16_train/', n_slice=9)
    # Test dataset
    ds_test = GraphDataset(os.path.join(path_dataset, test_data['name_data']),
                           os.path.join(path_dataset, test_data['name_weights']))

    sampler_train = torch.utils.data.WeightedRandomSampler(ds_train.weights, len(ds_train.weights), replacement=True)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, train['batch_size'], drop_last=True)
    sampler_test = torch.utils.data.WeightedRandomSampler(ds_test.weights, len(ds_test.weights), replacement=True)
    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, train['batch_size'], drop_last=True)

    # collate all examples in one batch
    collate_fn = functools.partial(collate, node_feature_dims=node_feature_mapping_dim,
                                   edge_feature_dims=edge_feature_mapping_dim, lMax=lMax,
                                   prop_max_edges_given=train['prop_max_edges_given'])

    # Generate a DataLoader
    dataloader_train = torch.utils.data.DataLoader(
        ds_train,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=train_data['num_workers'])
    dataloader_test = torch.utils.data.DataLoader(
        ds_test,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler_test,
        pin_memory=True,
        num_workers=test_data['num_workers'])

    # Generate weight_types matrix ?????
    # TODO: EDGE_IDX_MAP must be a parameter into the config file
    weight_types = torch.ones(len(EDGE_IDX_MAP), dtype=torch.float, device=set_cuda_status(cpu, Logger))
    for k, weight_type in weight_types_d.items():
        weight_types[EDGE_IDX_MAP[k]] = weight_type

    # Model initialization
    model = GravTransformer(node_feature_mapping_dim,
                            edge_feature_mapping_dim,
                            arch['embedding_dim'],
                            arch['n_head'],
                            arch['num_layers'],
                            arch['positional_encoding'],
                            lMax)
    model.to(device)

    # Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=train['lr'])
    if train['scheduler_lr']:
        scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=train['scheduler_step'])
        # scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimiser,
        #    partial(schedule, num_epochs=train['num_epochs']))
    exp_logger.reset_meters('hyperparams')
    learning_rate = optimizer.param_groups[0]['lr']
    exp_logger.update_value_meter('hyperparams', 'learning_rate', learning_rate)

    # Set output strings
    output_loss = "epoch, loss\n"
    output_precision = "epoch, ±, " + ", ".join([k.name for k in EDGE_IDX_MAP.keys()]) + '\n'
    output_recall = "epoch, ±, " + ", ".join([k.name for k in EDGE_IDX_MAP.keys()]) + '\n'

    for epoch in range(train['num_epochs']):
        Logger.info("epoch {}\n".format(epoch + 1))
        exp_logger.reset_meters('train')
        model.train(True)
        for n, data in enumerate(dataloader_train):
            data.load_cuda_async(device=device)
            prediction = model(data)
            loss = GravTransformer.loss(prediction, data, coef_neg=coef_neg, weight_types=weight_types)
            exp_logger.update_meter('train', 'loss', loss.item(), n=1)
            # exp_logger.run["train/loss"].log(loss.item())
            if n % train['print_freq'] == 0:
                perf_edge, perf_type = GravTransformer.performances(prediction, data)
                # print(perf_edge)
                #print("type :", perf_type)
                exp_logger.update_meter('train', 'perf_edge', np.mean(perf_edge), n=1)
                exp_logger.update_meter('train', 'perf_type', metrics.agreg_performances(perf_type)[0], n=1)
                #exp_logger.run["train/perf_edge"].log(exp_logger.get_meter('train', 'perf_edge'))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'LR {lr:.2e}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Perf edge {pedge.avg:.3f} ({pedge.val:.3f})\t'
                      'Perf types {ptype.avg:.3f} ({ptype.val:.3f})'.format(
                          epoch, n, len(dataloader_train), lr=train['lr'],
                          loss=exp_logger.get_meter('train', 'loss'), pedge=exp_logger.get_meter('train', 'perf_edge'),
                          ptype=exp_logger.get_meter('train', 'perf_type')))
            # if not n%(len(dataloader_train)//5):
            #    print("{:.0%}  {:.3f}\n".format(n/len(dataloader_train), loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        exp_logger.log_meters('train', n=epoch)
        exp_logger.log_meters('hyperparams', n=epoch)

        model.train(False)
        exp_logger.reset_meters('test')
        losses = []
        perfs_edge, perfs_type = [], []
        for n, data in enumerate(dataloader_test):
            if test_data['num_batch_eval']:
                if n >= test_data['num_batch_eval']:
                    break
            data.load_cuda_async(device=device)
            with torch.no_grad():
                prediction = model(data)
                loss = GravTransformer.loss(prediction, data, coef_neg=coef_neg, weight_types=weight_types).item()
                exp_logger.update_meter('test', 'loss', loss, n=1)
                losses.append(loss)
            perfs = GravTransformer.performances(prediction, data)
            perfs_edge.append(perfs[0])
            perfs_type.append(perfs[1])
        loss_avg = np.mean(losses)
        output_loss += "{}, {:.3f}\n".format(epoch, loss_avg)
        precision_recall = metrics.agreg_performances(perfs_edge)
        output_precision += "{}, {:.2f}".format(epoch, precision_recall[0])
        output_recall += "{}, {:.2f}".format(epoch, precision_recall[1])
        precision_recall_type = metrics.agreg_performances(perfs_type)
        for k, i in EDGE_IDX_MAP.items():
            output_precision += ", {:.2f}".format(precision_recall_type[0][i])
            output_recall += ", {:.2f}".format(precision_recall_type[1][i])

        # with open(args['f_output']+"output_loss.log", "at") as f:
        #    f.write(output_loss)
        #    output_loss = ""
        # with open(args['f_output']+"output_precision.log", "at") as f:
        #    f.write(output_precision+'\n')
        #    output_precision = ""
        # with open(args['f_output']+"output_recall.log", "at") as f:
        #    f.write(output_recall+'\n')
        #    output_recall = ""
        exp_logger.log_meters('test', n=epoch)

        if (epoch and not epoch % 5) or epoch == train['num_epochs'] - 1:
            torch.save(model.state_dict(), train['f_output'] + "model-ep{}.pt".format(epoch))
            torch.save(optimizer.state_dict(), train['f_output'] + "optimiser-ep{}.pt".format(epoch))

        if train['scheduler_lr']:
            scheduler_lr.step(loss_avg)


@ex.automain
def main():
    pass

# if __name__ == '__main__':
#    main()
