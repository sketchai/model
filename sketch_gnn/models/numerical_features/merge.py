import torch


class NumericalFeaturesMerge(torch.nn.Module):
    """Transform a sequence of numerical feature vectors into a single vector.

    Currently, this module simply aggregates the features by averaging, although more
    elaborate aggregation schemes (e.g. RNN) could be chosen.

    ----------------------------
    Source : SketchGraph model. sketchgraphs_models/graph/model/numerical_features.py
    """

    def __init__(self):
        super(NumericalFeaturesMerge, self).__init__()

    def forward(self, embeddings):
        return embeddings.sum(axis=-2)
