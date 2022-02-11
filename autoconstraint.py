import gzip
import pickle
import functools
import argparse

import numpy as np
import torch

from sketchgraphs.data import flat_array, sequence

from models.dense_emb import *
from models.transf_graph import *
from loaders.graph_data import *


n_head = 4
num_layers = 2
embedding_dim = 60 * n_head
batch_size = 750
num_epochs = 10
lr = 5e-4
schedule_lr = False

prop_max_edges_given = 0.1
positional_encoding = True

num_batch_eval = 20
num_workers = 8
f_output = "output/"

coef_neg = 6
weight_types_d = {}  # sequence.ConstraintType.Vertical: 4,
#                sequence.ConstraintType.Perpendicular: 3,
#                sequence.ConstraintType.Length: 2.5,
#                sequence.ConstraintType.Diameter: 4}


def schedule(epoch):
    """
    Apply a multiplicative coefficient to lr.
    """
    if epoch < 2 * num_epochs // 5:
        return 3**(epoch / (num_epochs // 5) - 2)
    elif epoch > 3 * num_epochs // 5:
        return 10**(3 - epoch / (num_epochs // 5))
    else:
        return 1


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_output', type=str, default=f_output, help='Output directory.')

    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--lr', type=float, default=lr)

    parser.add_argument('--embedding_dim', type=int, default=embedding_dim)
    parser.add_argument('--n_head', type=int, default=n_head)
    parser.add_argument('--num_layers', type=int, default=num_layers)
    parser.add_argument('--positional_encoding', dest='positional_encoding', action='store_true')
    parser.add_argument('--no_positional_encoding', dest='positional_encoding', action='store_false')
    parser.set_defaults(positional_encoding=positional_encoding)

    parser.add_argument('--prop_max_edges_given', type=float, default=prop_max_edges_given,
                        help="Maximal proportion of edges given in training examples.")

    parser.add_argument('--num_batch_eval', type=int, default=num_batch_eval,
                        help="Number of batches over which evaluation is averaged. Set to 0 to average over the whole validation set.")
    parser.add_argument('--num_workers', type=int, default=num_workers,
                        help="Number of CPUs.")

    return parser


def main():
    argparser = create_argparser()
    args = vars(argparser.parse_args())
    output_params = ""
    for k, e in args.items():
        output_params += "{} = {}\n".format(k, e)

    # modify the following line
    with open('data/sg_t16_train/preprocessing_params.pkl', 'rb') as f:
        preprocessing_params = pickle.load(f)
    lMax = preprocessing_params['lMax']
    node_feature_mapping_dim = preprocessing_params['node_feature_dimensions']
    edge_feature_mapping_dim = preprocessing_params['edge_feature_dimensions']
    output_params += '\n'
    for k, e in preprocessing_params.items():
        output_params += "{} = {}\n".format(k, e)

    with open(args['f_output'] + "parameters.txt", "wt") as f:
        f.write(output_params)

    device = torch.device('cuda')

    # modify the following lines
    ds_train = GraphDataset('data/sg_t16_train/',
                            'data/sg_t16_train/', n_slice=9)
    ds_test = GraphDataset('data/sg_t16_validation_final.npy',
                           'data/sg_t16_validation_weights.npy')

    sampler_train = torch.utils.data.WeightedRandomSampler(ds_train.weights, len(ds_train.weights), replacement=True)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args['batch_size'], drop_last=True)
    sampler_test = torch.utils.data.WeightedRandomSampler(ds_test.weights, len(ds_test.weights), replacement=True)
    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args['batch_size'], drop_last=True)

    collate_fn = functools.partial(collate, node_feature_dims=node_feature_mapping_dim, edge_feature_dims=edge_feature_mapping_dim, lMax=lMax, prop_max_edges_given=args['prop_max_edges_given'])

    dataloader_train = torch.utils.data.DataLoader(
        ds_train,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=args['num_workers'])
    dataloader_test = torch.utils.data.DataLoader(
        ds_test,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler_test,
        pin_memory=True,
        num_workers=args['num_workers'])

    weight_types = torch.ones(len(EDGE_IDX_MAP), dtype=torch.float, device=device)
    for k, weight_type in weight_types_d.items():
        weight_types[EDGE_IDX_MAP[k]] = weight_type

    model = GravTransformer(node_feature_mapping_dim, edge_feature_mapping_dim, args['embedding_dim'], args['n_head'], args['num_layers'], args['positional_encoding'], lMax)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args['lr'])
    if schedule_lr:
        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)

    output_loss = "epoch, loss\n"
    output_precision = "epoch, ±, " + ", ".join([k.name for k in EDGE_IDX_MAP.keys()]) + '\n'
    output_recall = "epoch, ±, " + ", ".join([k.name for k in EDGE_IDX_MAP.keys()]) + '\n'

    for epoch in range(args['num_epochs']):
        print("epoch {}\n".format(epoch + 1))

        model.train(True)
        for n, data in enumerate(dataloader_train):
            data.load_cuda_async(device=device)
            prediction = model(data)
            loss = GravTransformer.loss(prediction, data, coef_neg=coef_neg, weight_types=weight_types)
            if not n % (len(dataloader_train) // 5):
                print("{:.0%}  {:.3f}\n".format(n / len(dataloader_train), loss.item()))
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        model.train(False)
        losses = []
        perfs_edge, perfs_type = [], []
        for n, data in enumerate(dataloader_test):
            if num_batch_eval:
                if n >= num_batch_eval:
                    break
            data.load_cuda_async(device=device)
            with torch.no_grad():
                prediction = model(data)
                losses.append(GravTransformer.loss(prediction, data, coef_neg=coef_neg, weight_types=weight_types).item())
            perfs = GravTransformer.performances(prediction, data)
            perfs_edge.append(perfs[0])
            perfs_type.append(perfs[1])
        output_loss += "{}, {:.3f}\n".format(epoch, np.mean(losses))
        precision_recall = GravTransformer.agreg_performances(perfs_edge)
        output_precision += "{}, {:.2f}".format(epoch, precision_recall[0])
        output_recall += "{}, {:.2f}".format(epoch, precision_recall[1])
        precision_recall_type = GravTransformer.agreg_performances(perfs_type)
        for k, i in EDGE_IDX_MAP.items():
            output_precision += ", {:.2f}".format(precision_recall_type[0][i])
            output_recall += ", {:.2f}".format(precision_recall_type[1][i])

        with open(args['f_output'] + "output_loss.log", "at") as f:
            f.write(output_loss)
            output_loss = ""
        with open(args['f_output'] + "output_precision.log", "at") as f:
            f.write(output_precision + '\n')
            output_precision = ""
        with open(args['f_output'] + "output_recall.log", "at") as f:
            f.write(output_recall + '\n')
            output_recall = ""

        if (epoch and not epoch % 5) or epoch == args['num_epochs'] - 1:
            torch.save(model.state_dict(), args['f_output'] + "model-ep{}.pt".format(epoch))
            torch.save(optimiser.state_dict(), args['f_output'] + "optimiser-ep{}.pt".format(epoch))

        if schedule_lr:
            scheduler_lr.step()


if __name__ == '__main__':
    main()
