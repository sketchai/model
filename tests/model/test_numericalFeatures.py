import unittest
import logging
import torch
import enum

from sketch_gnn.models.numerical_features.generator import generate_embedding
from sketch_gnn.models.numerical_features.embedding import NumericalFeatureEmbedding

logger = logging.getLogger(__name__)


class TestNumericalFeatures(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.n_bins = 50
        self.d_features_dims = {'Angle': {'aligned': 3, 'clockwise': 3, 'angle': self.n_bins},  # example for Angle
                                'Diameter': {'length': self.n_bins},  # Diameter
                                # 'Point': {'statusconstruction': 2, 'x': 3, 'y': 3},
                                }

    def test_numerical_feature_encoding(self):
        l_values = self.d_features_dims.get('Angle').values()
        self.assertListEqual(list(l_values), [3, 3, self.n_bins])

        feature_encoder = NumericalFeatureEmbedding(feature_dims=l_values, embedding_dim=4, n_max_params=4)
        # Check information about the embeddings attribut (matrix of size (self.n_bins +6,4) )
        self.assertEqual(feature_encoder.embeddings.embedding_dim, 4)
        self.assertEqual(feature_encoder.embeddings.num_embeddings, self.n_bins + 3 + 3 + 1)

        # Check the information about the offset : not clear how the offset was chosen
        torch.testing.assert_allclose(feature_encoder.feature_offsets, torch.tensor([0, 3, 6, 56]))

        logger.info(f'feature = {feature_encoder}')

        in_tensor = torch.tensor([1,2,3,0], dtype=torch.int64)
        out_tensor = feature_encoder(in_tensor)
        logger.debug(out_tensor)

    def test_generate_embedding(self):

        res = generate_embedding(d_features_dims=self.d_features_dims, embedding_dim=4)

        # Test 1: Check that the embedding is a dict which keys are string
        self.assertListEqual(list(res.keys()), ['Angle', 'Diameter'])

        # Test 2: Check the construction of the associated embeddings
        angle_embedding = res.get('Angle')
        named_layers = dict(angle_embedding.named_modules())
        self.assertListEqual(list(named_layers.keys()), ['', '0', '0.embeddings', '1'])

        # for key in ['', '0', '0.embeddings', '1']:
        #     self.assertTrue(torch.equal(torch.nn.embedding()))

