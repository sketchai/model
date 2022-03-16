from src.dataloader.batch_data import GraBatch

def collate(batch, node_feature_dims, edge_feature_dims, lMax, prop_max_edges_given=0.9, generation=False, mask_attention=True):
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
    node_features = []
    sparse_node_features = {k: {'index': [], 'value': []} for k in node_feature_dims.keys()}
    edge_features = []
    sparse_edge_features = {k: {'index': [], 'value': []} for k in edge_feature_dims.keys()}
    incidences = []
    edges_toInf_pos = []
    edges_toInf_pos_types = []
    edges_toInf_neg = []
    src_key_padding_mask = []
    is_given = []

    for n, ex in enumerate(batch):
        node_features.append(ex['node_features'])
        for k in node_feature_dims.keys():
            sparse_node_features[k]['index'].append(ex['sparse_node_features'][k]['index'] + n * lMax)
            sparse_node_features[k]['value'].append(ex['sparse_node_features'][k]['value'])

        l = len(ex['i_edges_possible'])
        n_max_edges_given = min(int(prop_max_edges_given * l), l - 2)
        if l > 2:
            i_given = RNG.choice(ex['i_edges_possible'], int(RNG.uniform(0, n_max_edges_given)), replace=False)
        else:
            i_given = np.array([], dtype=np.int64)
        i_given = np.concatenate([i_given, ex['i_edges_given']])
        is_given.append(i_given)
        maskCompl = np.ones(len(ex['incidences']), dtype=bool)
        maskCompl[i_given] = False

        incidences.append(ex['incidences'][i_given] + n * lMax)
        edge_features.append(ex['edge_features'][i_given])

        i_given = torch.tensor(i_given).unsqueeze(0)
        for k in edge_feature_dims.keys():
            i_given_sparse_features = torch.nonzero(i_given - ex['sparse_edge_features'][k]['index'].unsqueeze(1) == 0,
                                                    as_tuple=True)[0]  # indices of ex['sparse_edge_features']['index'] that are in i_given
            sparse_edge_features[k]['value'].append(ex['sparse_edge_features'][k]['value'][i_given_sparse_features])
        edges_toInf_pos.append(ex['incidences'][maskCompl] + n * lMax)
        edges_toInf_pos_types.append(ex['edge_features'][maskCompl])

        edges_toInf_neg.append(ex['edges_toInf_neg'] + n * lMax)
        src_key_padding_mask.append(ex['mask_attention'])

    node_features = torch.cat(node_features)
    for k in node_feature_dims.keys():
        sparse_node_features[k]['index'] = torch.cat(sparse_node_features[k]['index'])
        sparse_node_features[k]['value'] = torch.vstack(sparse_node_features[k]['value'])
    incidences = torch.vstack(incidences).T.contiguous()
    incidences = torch.cat((incidences, torch.flip(incidences, [0])), dim=1)  # non-oriented graph, symmetrize
    edge_features = torch.cat(edge_features)
    edge_features = edge_features.repeat(2)
    for k in sparse_edge_features.keys():
        sparse_edge_features[k]['index'] = torch.nonzero(
            edge_features == EDGE_IDX_MAP.get(k, -1), as_tuple=True)[0]
        sparse_edge_features[k]['value'] = torch.vstack(sparse_edge_features[k]['value'])
        sparse_edge_features[k]['value'] = sparse_edge_features[k]['value'].repeat(2, 1)
    if not generation:
        edges_toInf_pos = torch.vstack(edges_toInf_pos).contiguous()
        edges_toInf_pos_types = torch.cat(edges_toInf_pos_types).contiguous()
        edges_toInf_neg = torch.vstack(edges_toInf_neg)
    else:  # no evaluation then
        edges_toInf_pos = torch.vstack(edges_toInf_pos + edges_toInf_neg).contiguous()
        edges_toInf_pos_types = torch.empty((0,), dtype=torch.int64).contiguous()
        edges_toInf_neg = torch.empty((0, 2), dtype=torch.int64)
    src_key_padding_mask = torch.vstack(src_key_padding_mask)

    positions = torch.arange(lMax)

    return GraBatch({
        'l_batch': len(batch),
        'node_features': node_features,
        'sparse_node_features': sparse_node_features,
        'incidences': incidences,
        'edge_features': edge_features,
        'sparse_edge_features': sparse_edge_features,
        'edges_toInf_pos': edges_toInf_pos,
        'edges_toInf_pos_types': edges_toInf_pos_types,
        'edges_toInf_neg': edges_toInf_neg,
        'src_key_padding_mask': src_key_padding_mask if mask_attention else None,
        'positions': positions,
        'is_given': is_given if generation else None  # np.ndarray cannot be moved to gpu
    })
