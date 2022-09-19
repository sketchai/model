import unittest
import logging
import torch

from sketch_gnn.models.concatenatelinear import ConcatenateLinear

logger = logging.getLogger(__name__)


class TestConcatenateLinear(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.concatenate = ConcatenateLinear(left_size=10, right_size=1, output_size=5)

    def test_forward(self):
        right_tensor = torch.ones(2, 8)
        left_tensor = torch.zeros(2, 3)

        # Check size of A and b
        A = self.concatenate._linear.weight  # can change, never fix
        self.assertEqual(A.size(), (5, 11))
        b = self.concatenate._linear.bias  # can change, never fix
        self.assertEqual(b.size(), (5,))

        # Generate the new vector
        vect = self.concatenate.forward(left_tensor, right_tensor)

        # Reconstruct it with another method
        # Torch cat = concatenation with respect to the last dimension
        x = torch.cat((left_tensor, right_tensor), dim=-1)
        ground_truth = torch.tensor([[0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                                     [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.]])
        self.assertTrue(torch.equal(x, ground_truth))
        self.assertEqual(x.size(), (2, 11))

        # Apply the multiplication and addition
        self.assertEqual(A.transpose(dim0=0, dim1=1).size(), (11, 5))
        value = torch.matmul(x, A.transpose(dim0=0, dim1=1)) + b
        self.assertTrue(torch.equal(vect, value))
