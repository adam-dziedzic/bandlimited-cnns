import numpy as np
import torch
import unittest
from torch import tensor

from cnns.nnlib.layers import conv_backward_naive_1D
from cnns.nnlib.layers import conv_forward_naive_1D
from cnns.nnlib.pytorch_layers.pytorch_conv1D_fft \
    import PyTorchConv1dFunction, PyTorchConv1dAutograd


class TestPyTorchConv1d(unittest.TestCase):

    def test_FunctionForwardNoCompression(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :],
                                       mode="valid")
        conv = PyTorchConv1dFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_FunctionBackwardNoCompression(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        x_torch = tensor(x, requires_grad=True)
        y_torch = tensor(y, requires_grad=True)
        b_torch = tensor(b, requires_grad=True)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b,
                                                       conv_param)

        result_torch = PyTorchConv1dFunction.apply(x_torch, y_torch,
                                             b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2]]])
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        dx, dw, db = result.backward(dout)

        # are the gradients correct
        np.testing.assert_array_almost_equal(dx, expected_dx)
        np.testing.assert_array_almost_equal(dw, expected_dw)
        np.testing.assert_array_almost_equal(db, expected_db)

    def test_FunctionForwardWithCompression(self):
        # test with compression
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        expected_result = [3.5, 7.5]
        conv_param = {'preserve_energy_rate': 0.9}
        conv = PyTorchConv1dFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b),
                              preserve_energy_rate=conv_param[
                                  'preserve_energy_rate'])
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_AutogradForwardNoCompression(self):
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :],
                                       mode="valid")
        conv = PyTorchConv1dAutograd(filter=torch.from_numpy(y),
                                     bias=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))

        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_AutogradForwardWithCompression(self):
        # test with compression
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        expected_result = [3.5, 7.5]
        conv_param = {'preserve_energy_rate': 0.9}
        conv = PyTorchConv1dAutograd(
            filter=torch.from_numpy(y), bias=torch.from_numpy(b),
            preserve_energy_rate=conv_param[
                'preserve_energy_rate'])
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))


if __name__ == '__main__':
    unittest.main()
