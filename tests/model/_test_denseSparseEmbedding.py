import unittest
import logging
import torch

from src.models.dense_emb import DenseSparsePreEmbedding

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class TestDenseSparseEmbedding(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.denseSparseEmbedding = DenseSparsePreEmbedding(left_size=10, right_size=1, output_size=5)

    def test_forward(self):
