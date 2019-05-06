from cnns.nnlib.robustness.disk_mask import get_complex_mask

import torch
import unittest
import numpy as np


class TestGetComplexMask(unittest.TestCase):

    def test_get_comples_mask(self):
        mask = get_complex_mask(side_len=7, compress_rate=26, val=0)
        print("mask: ", mask)
        desired = torch.tensor([[[1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.],
         [0., 0.],
         [1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [0., 0.],
         [0., 0.],
         [0., 0.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.],
         [0., 0.],
         [1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.]]])
        np.testing.assert_equal(actual=mask.numpy(), desired=desired.numpy())