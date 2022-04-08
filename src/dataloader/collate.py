import numpy as np
import torch

from src.dataloader.batch_data import GraBatch

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

RNG = np.random.default_rng()


def collect_batch(batch, node_elements, edge_elements, lMax, prop_max_edges_given):
    batch_data = lambda : None
    batch_data.node_features = []
    batch_data.sparse_node_features = {k: {'index': [], 'value': []} for k in node_elements}
    batch_data.sparse_edge_features = {k: {'index': [], 'value': []} for k in edge_elements}
    batch_data.incidences = []
    batch_data.edges_toInf_pos = []
    batch_data.edge_features = []
    batch_data.edges_toInf_pos_types = []
    batch_data.edges_toInf_neg = []
    batch_data.src_key_padding_mask = []
    batch_data.given_index_edges = []

    for n, ex in enumerate(batch):

        shift = n * lMax

        # node_features
        batch_data.node_features.append(ex['node_features'])

        # node mask attention
        batch_data.src_key_padding_mask.append(ex['mask_attention'])

        # sparse_node_features 
        for k in node_elements :
            batch_data.sparse_node_features[k]['index'].append(ex['sparse_node_features'][k]['index'] + shift) # shift ?
            batch_data.sparse_node_features[k]['value'].append(ex['sparse_node_features'][k]['value'])

        # Prepare a subgraph of constraints : given index edges are selected randomly among the constraint list
        l = len(ex['i_edges_possible']) # compute the number of subnode constraints on the current ex 
        n_max_edges_given = min(int(prop_max_edges_given * l), l - 2)
        if l > 2:
            curr_given_index_edges = RNG.choice(ex['i_edges_possible'], int(RNG.uniform(0, n_max_edges_given)), replace=False)
        else:
            curr_given_index_edges = np.array([], dtype=np.int64)
        curr_given_index_edges = np.concatenate([curr_given_index_edges, ex['i_edges_given']])
        batch_data.given_index_edges.append(curr_given_index_edges)

        batch_data.incidences.append(ex['incidences'][curr_given_index_edges] + shift) # pourquoi ne pas avoir une vrai matrice d'incidences ? 
        batch_data.edge_features.append(ex['edge_features'][curr_given_index_edges])

        curr_given_index_edges = torch.tensor(curr_given_index_edges).unsqueeze(0)
        for k in edge_elements:
            i_given_sparse_features = torch.nonzero(curr_given_index_edges - ex['sparse_edge_features'][k]['index'].unsqueeze(1) == 0,
                                                    as_tuple=True)[0]  # indices of ex['sparse_edge_features']['index'] that are in i_given
            batch_data.sparse_edge_features[k]['value'].append(ex['sparse_edge_features'][k]['value'][i_given_sparse_features])

        # Mask to represent unknown connections
        nb_edges_connection = len(ex['incidences'])
        maskCompl = np.ones(nb_edges_connection, dtype=bool)
        maskCompl[curr_given_index_edges] = False

        batch_data.edges_toInf_pos.append(ex['incidences'][maskCompl] + shift)
        batch_data.edges_toInf_pos_types.append(ex['edge_features'][maskCompl]) 
        batch_data.edges_toInf_neg.append(ex['edges_toInf_neg'] + shift)
        
    return batch_data


def collate(batch, node_feature_dims, edge_feature_dims, edge_idx_map,  lMax, prop_max_edges_given=0.9, generation=False, mask_attention=True):
    """
    Function to collate examples in one batch.
    batch: list of examples;
    node_feature_dims: dictionary {primitive: {feature: dimension}}, as returned by the preprocessing in 'preprocessing_params.pkl';
    edge_feature_dims: dictionary {constraint: {feature: dimension}}, as returned by the preprocessing in 'preprocessing_params.pkl';
    lMax: int, length of the examples, returned by the preprocessing in 'preprocessing_params.pkl';
    prop_max_edges_given: float, maximal proportion of edges of the example that are given to the neural network. For each example, a proportion p ~ uniform(0, prop_max_edges_given) of edges are given, among the possible ones. No inference is done on these;
    generation: bool, set to False for training, to True for using the trained neural network;
    mask_attention: bool, to generate a mask on the padding nodes for the attention mecanism.
    """

    batch_data = collect_batch(batch, node_feature_dims.keys(), edge_feature_dims.keys(),lMax, prop_max_edges_given)

    batch_data.node_features = torch.cat(batch_data.node_features)
    for key in batch_data.sparse_node_features.keys():
        batch_data.sparse_node_features[key]['index'] = torch.cat(batch_data.sparse_node_features[key]['index'])
        batch_data.sparse_node_features[key]['value'] = torch.vstack(batch_data.sparse_node_features[key]['value'])

    batch_data.edge_features = torch.cat(batch_data.edge_features)
    batch_data.edge_features = batch_data.edge_features.repeat(2)

    for key in batch_data.sparse_edge_features.keys():
        batch_data.sparse_edge_features[key]['index'] = torch.nonzero(
                                batch_data.edge_features == edge_idx_map.get(key, -1), as_tuple=True)[0]
        batch_data.sparse_edge_features[key]['value'] = torch.vstack(batch_data.sparse_edge_features[key]['value'])
        batch_data.sparse_edge_features[key]['value'] = batch_data.sparse_edge_features[key]['value'].repeat(2, 1)

    batch_data.incidences = torch.vstack(batch_data.incidences).T.contiguous()
    batch_data.incidences = torch.cat((batch_data.incidences, torch.flip(batch_data.incidences, [0])), dim=1)  # non-oriented graph, symmetrize

    if not generation: #if not training
        batch_data.edges_toInf_pos = torch.vstack(batch_data.edges_toInf_pos).contiguous()
        batch_data.edges_toInf_pos_types = torch.cat(batch_data.edges_toInf_pos_types).contiguous()
        batch_data.edges_toInf_neg = torch.vstack(batch_data.edges_toInf_neg)
    else:  # no evaluation then
        batch_data.edges_toInf_pos = torch.vstack(batch_data.edges_toInf_pos + batch_data.edges_toInf_neg).contiguous()
        batch_data.edges_toInf_pos_types = torch.empty((0,), dtype=torch.int64).contiguous()
        batch_data.edges_toInf_neg = torch.empty((0, 2), dtype=torch.int64)


    #batch_data.l_batch = len(batch)
    #batch_data.given_index_edges = batch_data.given_index_edges if generation else None
    #batch_data.positions = torch.arange(lMax)
    #batch_data.src_key_padding_mask = torch.vstack(batch_data.src_key_padding_mask) if mask_attention else None

    return GraBatch(
        {
        'l_batch': len(batch),
        'node_features': batch_data.node_features,
        'sparse_node_features': batch_data.sparse_node_features,
        'incidences': batch_data.incidences,
        'edge_features': batch_data.edge_features,
        'sparse_edge_features': batch_data.sparse_edge_features,
        'edges_toInf_pos': batch_data.edges_toInf_pos,
        'edges_toInf_pos_types': batch_data.edges_toInf_pos_types,
        'edges_toInf_neg': batch_data.edges_toInf_neg,
        'src_key_padding_mask': torch.vstack(batch_data.src_key_padding_mask) if mask_attention else None,
        'positions': torch.arange(lMax),
        'is_given': batch_data.given_index_edges if generation else None  # np.ndarray cannot be moved to gpu
    })
