from cnns.nnlib.utils.shift_DC_component import shift_DC
from cnns.nnlib.utils.shift_DC_component import shift_DC_elemwise

import torch
import unittest
from unittest import TestCase
import numpy as np
from numpy.testing import assert_equal


class TestShiftDCComponent(TestCase):

    def setUp(self) -> None:
        self.dtype = np.float32

    def test_one(self):
        xfft = torch.tensor([[[1.0]]])
        xfft_out = shift_DC(xfft)
        assert_equal(actual=xfft_out.numpy(), desired=np.array([[[1.0]]]))

    def test_single_element(self):
        value = 14.0
        xfft = torch.tensor([[[value]]])
        xfft_out = shift_DC(xfft)
        assert_equal(actual=xfft_out.numpy(),
                     desired=np.array([[[value]]], dtype=self.dtype))

    def test_two(self):
        val1 = 14.0
        val2 = 9.2
        xfft = torch.tensor([[[val1, val2]]])
        xfft_out = shift_DC(xfft)
        assert_equal(actual=xfft_out.numpy(),
                     desired=np.array([[[val1, val2]]], dtype=self.dtype))

    def test_two_bothsided(self):
        val1 = 14.0
        val2 = 9.2
        xfft = torch.tensor([[[val1, val2]]])
        xfft_out = shift_DC(xfft, onesided=False)
        assert_equal(actual=xfft_out.numpy(),
                     desired=np.array([[[val1, val2]]], dtype=self.dtype))

    def test_to_center_arange_9(self):
        xfft = np.arange(9, dtype=self.dtype).reshape(1, 3, 3, 1)
        xfft = torch.tensor(xfft)
        xfft_out = shift_DC(xfft, onesided=True)
        assert_equal(actual=xfft_out.numpy().squeeze(),
                     desired=np.array([[6., 7., 8.],
                                       [0., 1., 2.],
                                       [3., 4., 5.]], dtype=self.dtype))

    def test_symmetry(self):
        for n in range(1, 10, 1):
            print("n: ", n)
            xfft = np.arange(n**2, dtype=self.dtype).reshape(1, n, n, 1)
            xfft = torch.tensor(xfft)
            print("xfft: ", xfft)
            xfft_out = shift_DC(xfft, onesided=True, shift_to="center")
            print("xfft_out: ", xfft_out)
            xfft_out2 = shift_DC(xfft_out, onesided=True, shift_to="corner")
            print("xfft_out2: ", xfft_out2)
            assert_equal(actual=xfft_out2.numpy(), desired=xfft)

    def test_to_center_arange_9_bothsided(self):
        xfft = np.arange(9, dtype=self.dtype).reshape(1, 3, 3, 1)
        xfft = torch.tensor(xfft)
        xfft_out = shift_DC(xfft, onesided=False)
        assert_equal(actual=xfft_out.numpy().squeeze(),
                     desired=np.array([[8., 6., 7.],
                                       [2., 0., 1.],
                                       [5., 3., 4.]], dtype=self.dtype))

    def test_to_corner_arange_9(self):
        xfft = np.arange(9, dtype=self.dtype).reshape(1, 3, 3, 1)
        xfft = torch.tensor(xfft)
        xfft_out = shift_DC(xfft, onesided=True, shift_to="corner")
        assert_equal(actual=xfft_out.numpy().squeeze(),
                     desired=np.array([[3., 4., 5.],
                                       [6., 7., 8.],
                                       [0., 1., 2.]], dtype=self.dtype))

    def test_to_corner_arange_9_bothsided(self):
        xfft = np.arange(9, dtype=self.dtype).reshape(1, 3, 3, 1)
        xfft = torch.tensor(xfft)
        xfft_out = shift_DC(xfft, onesided=False, shift_to="corner")
        assert_equal(actual=xfft_out.numpy().squeeze(),
                     desired=np.array([[4., 5., 3.],
                                       [7., 8., 6.],
                                       [1., 2., 0.]], dtype=self.dtype))

    def test_to_center_elemwise_9(self):
        xfft = np.arange(9, dtype=self.dtype).reshape(1, 3, 3, 1)
        xfft = torch.tensor(xfft)
        xfft_out = shift_DC_elemwise(xfft, onesided=True)
        assert_equal(actual=xfft_out.numpy().squeeze(),
                     desired=np.array([[6., 7., 8.],
                                       [0., 1., 2.],
                                       [3., 4., 5.]], dtype=self.dtype))

    def test_to_center_elemwise_9_bothsided(self):
        xfft = np.arange(9, dtype=self.dtype).reshape(1, 3, 3, 1)
        xfft = torch.tensor(xfft)
        xfft_out = shift_DC_elemwise(xfft, onesided=False)
        assert_equal(actual=xfft_out.numpy().squeeze(),
                     desired=np.array([[8., 6., 7.],
                                       [2., 0., 1.],
                                       [5., 3., 4.]], dtype=self.dtype))

    def test_to_corner_arange_9_elemwise(self):
        xfft = np.arange(9, dtype=self.dtype).reshape(1, 3, 3, 1)
        print("xfft: ", xfft)
        xfft = torch.tensor(xfft)
        xfft_out = shift_DC_elemwise(xfft, onesided=True)
        assert_equal(actual=xfft_out.numpy().squeeze(),
                     desired=np.array([[6., 7., 8.],
                                       [0., 1., 2.],
                                       [3., 4., 5.]], dtype=self.dtype))

    def test_to_corner_arange_9_bothsided_elemwise(self):
        xfft = np.arange(9, dtype=self.dtype).reshape(1, 3, 3, 1)
        print("xfft: ", xfft)
        xfft = torch.tensor(xfft)
        xfft_out = shift_DC_elemwise(xfft, onesided=False)
        assert_equal(actual=xfft_out.numpy().squeeze(),
                     desired=np.array([[8, 6., 7.],
                                       [2., 0, 1.],
                                       [5., 3., 4.]], dtype=self.dtype))

    def test_elemwise_general_random(self):
        xfft = np.random.rand(32, 32, 1) # simulate complex maps
        xfft = torch.from_numpy(xfft)
        xfft_out1 = shift_DC_elemwise(xfft, onesided=False)
        xfft_out2 = shift_DC(xfft, onesided=False)
        assert_equal(actual=xfft_out1, desired=xfft_out2)

if __name__ == '__main__':
    unittest.main()
