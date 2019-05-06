from cnns.nnlib.robustness.disk_mask import get_complex_mask

import torch
import unittest
import numpy as np


class TestGetComplexMask(unittest.TestCase):

    def test_get_complex_mask(self):
        mask, array_mask = get_complex_mask(side_len=7, compress_rate=26, val=0)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 0., 1., 1., 1.],
             [1., 1., 0., 0., 0., 1., 1.],
             [1., 0., 0., 0., 0., 0., 1.],
             [1., 1., 0., 0., 0., 1., 1.],
             [1., 1., 1., 0., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1.]])
        np.testing.assert_equal(actual=array_mask, desired=desired_array_mask)

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

    def test_get_complex_mask_linear(self):
        mask, array_mask = get_complex_mask(side_len=7, compress_rate=26, val=0,
                                            interpolate="linear")
        array_mask = torch.tensor(array_mask)
        print("array mask:\n", array_mask)
        # print("mask:\n", array_mask)
        desired_array_mask = np.array(
            [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.6713, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 0.6713, 0.3379, 0.6713, 1.0000, 1.0000],
             [1.0000, 0.6713, 0.3379, 0.0046, 0.3379, 0.6713, 1.0000],
             [1.0000, 1.0000, 0.6713, 0.3379, 0.6713, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.6713, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
        np.testing.assert_allclose(actual=array_mask.numpy(),
                                   desired=desired_array_mask, rtol=1e-02)

    def test_get_complex_mask_exponent(self):
        mask, array_mask = get_complex_mask(side_len=7, compress_rate=26, val=0,
                                            interpolate="exponent")
        array_mask = torch.tensor(array_mask)
        print("array mask:\n", )
        # print("mask:\n", array_mask)
        desired_array_mask = np.array(
            [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.3730, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 0.3730, 0.1372, 0.3730, 1.0000, 1.0000],
             [1.0000, 0.3730, 0.1372, 0.0505, 0.1372, 0.3730, 1.0000],
             [1.0000, 1.0000, 0.3730, 0.1372, 0.3730, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.3730, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
        np.testing.assert_allclose(actual=array_mask.numpy(),
                                   desired=desired_array_mask, rtol=1e-03)

    def test_get_complex_mask_log(self):
        mask, array_mask = get_complex_mask(side_len=7, compress_rate=26,
                                            val=0,
                                            interpolate="log")
        array_mask = torch.tensor(array_mask)
        print("array mask:\n")
        print("mask:\n", array_mask)
        desired_array_mask = np.array(
            [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.7671, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 0.7671, 0.4578, 0.7671, 1.0000, 1.0000],
             [1.0000, 0.7671, 0.4578, 0.0079, 0.4578, 0.7671, 1.0000],
             [1.0000, 1.0000, 0.7671, 0.4578, 0.7671, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.7671, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]
        )
        np.testing.assert_allclose(actual=array_mask.numpy(),
                                   desired=desired_array_mask, rtol=1e-02)
