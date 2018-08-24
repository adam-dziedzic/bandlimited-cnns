import unittest

import logging
import numpy as np
import torch
from torch import tensor

from cnns.nnlib.layers import conv_backward_naive_1D
from cnns.nnlib.layers import conv_forward_naive_1D
from cnns.nnlib.pytorch_layers.pytorch_conv1D_reuse_map_fft \
    import PyTorchConv1dFunction, PyTorchConv1dAutograd

from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging


class MockContext(object):

    def __init__(self):
        super(MockContext, self).__init__()
        self.args = None
        self.needs_input_grad = None

    def save_for_backward(self, *args):
        self.args = args

    @property
    def saved_tensors(self):
        return self.args

    def set_needs_input_grad(self, number_needed):
        self.needs_input_grad = [True for _ in range(number_needed)]


class TestPyTorchConv1d(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv1D_reuse_map_fft.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

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

    def test_FunctionForwardNoCompressionManySignalsOneChannel(self):
        x = np.array([[[1., -1., 0.]], [[1., 2., 3.]]])
        y = np.array([[[-2.0, 3.0]]])
        b = np.array([0.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = PyTorchConv1dFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardNoCompressionManySignalsOneFilterTwoChannels(self):
        x = np.array([[[1., 2., 3.], [4., 5., 6.]],
                      [[1., -1., 0.], [2., 5., 6.]]])
        y = np.array([[[0.0, 1.0], [-1.0, -1.0]]])
        b = np.array([0.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = PyTorchConv1dFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardNoCompression2Signals2Filters2Channels(self):
        x = np.array(
            [[[1., 2., 3.], [4., 5., 6.]], [[1., -1., 0.], [2., 5., 6.]]])
        y = np.array([[[2., 1.], [1., 3.]], [[0.0, 1.0], [-1.0, -1.0]]])
        b = np.array([1.0, 1.0])
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b,
                                                   conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = PyTorchConv1dFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardRandom(self):
        num_channels = 3
        num_data_points = 11
        num_values_data = 21
        num_values_filter = 5
        num_filters = 3
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, num_values_data)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, num_values_filter)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b, conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = PyTorchConv1dFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionBackwardNoCompressionWithBayes(self):
        x = np.array([[[1.0, 2.0, 3.0]]])
        y = np.array([[[2.0, 1.0]]])
        b = np.array([2.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b,
                                                       conv_param)
        ctx = MockContext()
        ctx.set_needs_input_grad(3)
        result_torch = PyTorchConv1dFunction.forward(ctx, x_torch,
                                                     y_torch, b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        dx, dw, db, _, _, _, _, _ = PyTorchConv1dFunction.backward(ctx, dout)

        self.logger.debug("expected dx: " + str(expected_dx))
        self.logger.debug("computed dx: " + str(dx))

        # are the gradients correct
        np.testing.assert_array_almost_equal(dx.detach().numpy(),
                                             expected_dx)
        np.testing.assert_array_almost_equal(dw.detach().numpy(),
                                             expected_dw)
        np.testing.assert_array_almost_equal(db.detach().numpy(),
                                             expected_db)

    def test_FunctionBackwardNoCompressionNoBias(self):
        x = np.array([[[1.0, 2.0, 3.0]]])
        y = np.array([[[2.0, 1.0]]])
        b = np.array([0.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b,
                                                       conv_param)

        result_torch = PyTorchConv1dFunction.apply(x_torch, y_torch,
                                                   b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        # are the gradients correct
        np.testing.assert_array_almost_equal(x_torch.grad,
                                             expected_dx)
        np.testing.assert_array_almost_equal(y_torch.grad,
                                             expected_dw)
        np.testing.assert_array_almost_equal(b_torch.grad,
                                             expected_db)

    def test_FunctionBackwardNoCompression2Channels(self):
        x = np.array([[[1.0, 2.0, 3.0], [-1.0, -3.0, 2.0]]])
        y = np.array([[[2.0, 1.0], [-2.0, 3.0]]])
        # still it is only a single filter but with 2 channels
        b = np.array([0.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)

        result_torch = PyTorchConv1dFunction.apply(x_torch, y_torch, b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

        dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        print()
        print("expected dx: " + str(expected_dx))
        print("computed dx: {}".format(x_torch.grad))

        print("expected dw: {}".format(expected_dw))
        print("computed dw: {}".format(y_torch.grad))
        #
        # self.logger.debug("expected db: ", expected_db)
        # self.logger.debug("computed db: ", b_torch.grad)

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x=expected_dx, y=x_torch.grad,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(y_torch.grad,
                                             expected_dw)
        np.testing.assert_array_almost_equal(b_torch.grad,
                                             expected_db)

    def test_FunctionForwardWithCompression(self):
        # test with compression
        x = np.array([[[1., 2., 3.]]])
        y = np.array([[[2., 1.]]])
        b = np.array([0.0])
        expected_result = [4.0, 7.0]
        conv = PyTorchConv1dFunction()
        result = conv.forward(
            ctx=None, input=torch.from_numpy(x), filter=torch.from_numpy(y),
            bias=torch.from_numpy(b), index_back=1)
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
        expected_result = [4.0, 7.0]
        conv = PyTorchConv1dAutograd(
            filter=torch.from_numpy(y), bias=torch.from_numpy(b), index_back=1)
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_FunctionForwardBackwardRandom(self):
        num_channels = 3
        num_data_points = 11
        num_values_data = 21
        num_values_filter = 5
        num_filters = 3
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, num_values_data)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, num_values_filter)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive_1D(x, y, b, conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)
        conv = PyTorchConv1dFunction()
        result_torch = conv.forward(ctx=None, input=x_torch, filter=y_torch,
                              bias=b_torch)
        result = result_torch.detach().numpy()
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)

        # dout = tensor(result/100.0, dtype=dtype)
        dout = torch.randn(result_torch.shape)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive_1D(dout.numpy(), cache)

        result_torch.backward(dout)

        # print()
        # print("expected dx: " + str(expected_dx))
        # print("computed dx: {}".format(x_torch.grad))
        #
        # print("expected dw: {}".format(expected_dw))
        # print("computed dw: {}".format(y_torch.grad))
        # self.logger.debug("expected db: ", expected_db)
        # self.logger.debug("computed db: ", b_torch.grad)

        # are the gradients correct
        np.testing.assert_array_almost_equal(
            x=expected_dx, y=x_torch.grad,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_dw, y=y_torch.grad, decimal=4,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_db, y=b_torch.grad, decimal=5,
            err_msg="Expected x is different from computed y.")


if __name__ == '__main__':
    unittest.main()
