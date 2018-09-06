import logging
import unittest

import numpy as np
from torch import tensor

from cnns.nnlib.pytorch_layers.pytorch_conv2D_reuse_map_fft \
    import PyTorchConv2dAutograd, PyTorchConv2dFunction
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging


class TestPyTorchConv2d(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv2D_reuse_map_fft.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def test_AutogradForwardNoCompression(self):
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        conv = PyTorchConv2dAutograd(filter=y, bias=b, index_back=0,
                                     use_next_power2=False)
        result = conv.forward(input=x)
        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal.")

    def test_FunctionForwardNoCompression(self):
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        conv = PyTorchConv2dFunction()

        result = conv.forward(ctx=None, input=x, filter=y, bias=b, index_back=0,
                              use_next_power2=False)

        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal.")

    def test_2_channels_2_filters(self):
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]],
                     [[1., 1., 2.], [2., 3., 1.], [2., -1., 3.]]]])
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]], [[-1.0, 2.0], [3.0, -2.0]]],
                    [[[-1.0, 1.0], [2.0, 3.0]], [[-2.0, 1.0], [1.0, -3.0]]]])
        b = tensor([0.0, 0.0])
        conv = PyTorchConv2dFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b, index_back=0)
        expect = np.array([[[[23.0, 32.0], [30., 4.]], [[11.0, 12.0],
                                                        [13.0, -11.0]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result, decimal=5,
            err_msg="The expected array x and computed y are not almost equal.")

    def test_bias(self):
        # A single 2D input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([-1.0])
        conv = PyTorchConv2dFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b, index_back=0)
        expect = np.array([[[[21.0, 21.0], [17., 13.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal.")

    def test_FunctionForwardCompression(self):
        # A single 2D input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        conv = PyTorchConv2dFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b, index_back=1,
                              use_next_power2=False)
        expect = np.array([[[[21.5, 22.0], [17.5, 13.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal.")

    def testExact2DConvWith3channels2filters(self):
        # This example is from Stanford CS231n course:
        # based on: http://cs231n.github.io/convolutional-networks/
        x = tensor(
            [[[
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ], [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
                [0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0],
                [0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ], [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0],
                [0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 2.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]]])
        y = tensor([[
            [[1.0, 0.0, 0.0],
             [-1.0, -1.0, 0.0],
             [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 1.0],
             [0.0, 1.0, -1.0],
             [-1.0, -1.0, 1.0]],
            [[-1.0, 0.0, 1.0],
             [0.0, 0.0, 1.0],
             [0.0, -1.0, -1.0]]],
            [
                [[1.0, 0.0, -1.0],
                 [-1.0, 0.0, 1.0],
                 [-1.0, 1.0, 0.0]],
                [[-1.0, -1.0, 0.0],
                 [0.0, -1.0, 1.0],
                 [1.0, 1.0, -1.0]],
                [[1.0, -1.0, 1.0],
                 [0.0, -1.0, -1.0],
                 [1.0, 1.0, -1.0]]],
        ])
        b = tensor([1.0, 0.0])
        conv = PyTorchConv2dFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b, index_back=0,
                              padding=0)
        expect = np.array(
            [[[[-2.0000e+00, -1.0000e+00, 1.0000e+00, -2.0000e+00, -3.0000e+00],
               [5.0000e+00, 2.0000e+00, -2.0000e+00, 1.0000e+00, -6.0000e+00],
               [-1.0000e+00, -4.0000e+00, 1.0000e+00, -1.0000e+00, -3.0000e+00],
               [1.7881e-07, 1.0000e+00, -7.0000e+00, -4.7684e-07, -3.0000e+00],
               [5.0000e+00, 1.0000e+00, 2.0000e+00, 1.0000e+00, -1.0000e+00]],
              [[-4.0000e+00, 1.0000e+00, -2.0000e+00, -2.0000e+00, 3.0000e+00],
               [-3.0000e+00, -2.0000e+00, -1.0000e+00, 4.0000e+00, -2.0000e+00],
               [5.0000e+00, 1.0000e+00, -3.0000e+00, -5.0000e+00, 2.0000e+00],
               [-3.0000e+00, -5.0000e+00, 2.0000e+00, -1.0000e+00, -3.0000e+00],
               [-6.0000e+00, -6.0000e+00, -1.0000e+00, 3.0000e+00,
                -8.0000e+00]]]]
        )
        np.testing.assert_array_almost_equal(
            x=expect, y=result, decimal=5,
            err_msg="The expected array x and computed y are not almost equal")

    def test_FunctionConv2DWithStride(self):
        # Test stride for 3 channels and 2 filters.
        # based on: http://cs231n.github.io/convolutional-networks/
        x = tensor(
            [[[
                [2.0, 0.0, 2.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 2.0],
                [0.0, 2.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 0.0, 2.0],
                [2.0, 0.0, 1.0, 1.0, 1.0],
            ], [
                [1.0, 1.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 2.0, 1.0],
                [0.0, 1.0, 2.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 2.0, 1.0],
                [2.0, 1.0, 2.0, 1.0, 2.0],
            ], [
                [1.0, 1.0, 2.0, 1.0, 2.0],
                [1.0, 2.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [1.0, 2.0, 2.0, 0.0, 2.0],
                [0.0, 2.0, 2.0, 0.0, 0.0],
            ]]])
        y = tensor([[
            [[1.0, 0.0, 0.0],
             [-1.0, -1.0, 0.0],
             [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 1.0],
             [0.0, 1.0, -1.0],
             [-1.0, -1.0, 1.0]],
            [[-1.0, 0.0, 1.0],
             [0.0, 0.0, 1.0],
             [0.0, -1.0, -1.0]]],
            [
                [[1.0, 0.0, -1.0],
                 [-1.0, 0.0, 1.0],
                 [-1.0, 1.0, 0.0]],
                [[-1.0, -1.0, 0.0],
                 [0.0, -1.0, 1.0],
                 [1.0, 1.0, -1.0]],
                [[1.0, -1.0, 1.0],
                 [0.0, -1.0, -1.0],
                 [1.0, 1.0, -1.0]]],
        ])
        b = tensor([1.0, 0.0])
        conv = PyTorchConv2dFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b, index_back=0,
                              padding=(1, 1), stride=(2, 2))
        expect = np.array([[[
            [-2.0, 1.0, -3.0],
            [-1.0, 1.0, -3.0],
            [5.0, 2.0, -1.0]], [
            [-4.0, -2.0, 3.0],
            [5.0, -3.0, 2.0],
            [-6.0, -1.0, -8.0]],
        ]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result, decimal=5,
            err_msg="The expected array x and computed y are not almost equal")

        # def test_FunctionForwardSpectralPooling(self):
        #     x = np.array([[[1., 2., 3., 4., 5., 6., 7., 8.]]])
        #     y = np.array([[[2., 1.]]])
        #     b = np.array([0.0])
        #     # get the expected results from numpy correlate
        #     expected_result = np.array(
        #         [[[2.771341, 5.15668, 9.354594, 14.419427]]])
        #     conv = PyTorchConv1dFunction()
        #     result = conv.forward(ctx=None, input=torch.from_numpy(x),
        #                           filter=torch.from_numpy(y),
        #                           bias=torch.from_numpy(b), out_size=4)
        #     np.testing.assert_array_almost_equal(
        #         x=np.array(expected_result), y=result,
        #         err_msg="Expected x is different from computed y.")
        #
        # def test_FunctionForwardNoCompressionManySignalsOneChannel(self):
        #     x = np.array([[[1., -1., 0.]], [[1., 2., 3.]]])
        #     y = np.array([[[-2.0, 3.0]]])
        #     b = np.array([0.0])
        #     # get the expected result
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, _ = conv_forward_naive_1D(x, y, b,
        #                                                conv_param)
        #     self.logger.debug("expected result: " + str(expected_result))
        #
        #     conv = PyTorchConv1dFunction()
        #     result = conv.forward(ctx=None, input=torch.from_numpy(x),
        #                           filter=torch.from_numpy(y),
        #                           bias=torch.from_numpy(b))
        #     self.logger.debug("obtained result: " + str(result))
        #     np.testing.assert_array_almost_equal(
        #         result, np.array(expected_result))
        #
        # def test_FunctionForwardNoCompressionManySignalsOneFilterTwoChannels(self):
        #     x = np.array([[[1., 2., 3.], [4., 5., 6.]],
        #                   [[1., -1., 0.], [2., 5., 6.]]])
        #     y = np.array([[[0.0, 1.0], [-1.0, -1.0]]])
        #     b = np.array([0.0])
        #     # get the expected result
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, _ = conv_forward_naive_1D(x, y, b,
        #                                                conv_param)
        #     self.logger.debug("expected result: " + str(expected_result))
        #
        #     conv = PyTorchConv1dFunction()
        #     result = conv.forward(ctx=None, input=torch.from_numpy(x),
        #                           filter=torch.from_numpy(y),
        #                           bias=torch.from_numpy(b))
        #     self.logger.debug("obtained result: " + str(result))
        #     np.testing.assert_array_almost_equal(
        #         result, np.array(expected_result))
        #
        # def test_FunctionForwardNoCompression2Signals2Filters2Channels(self):
        #     x = np.array(
        #         [[[1., 2., 3.], [4., 5., 6.]], [[1., -1., 0.], [2., 5., 6.]]])
        #     y = np.array([[[2., 1.], [1., 3.]], [[0.0, 1.0], [-1.0, -1.0]]])
        #     b = np.array([1.0, 1.0])
        #     # get the expected result
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, _ = conv_forward_naive_1D(x, y, b,
        #                                                conv_param)
        #     self.logger.debug("expected result: " + str(expected_result))
        #
        #     conv = PyTorchConv1dFunction()
        #     result = conv.forward(ctx=None, input=torch.from_numpy(x),
        #                           filter=torch.from_numpy(y),
        #                           bias=torch.from_numpy(b))
        #     self.logger.debug("obtained result: " + str(result))
        #     np.testing.assert_array_almost_equal(
        #         result, np.array(expected_result))
        #
        # def test_FunctionForwardRandom(self):
        #     num_channels = 3
        #     num_data_points = 11
        #     num_values_H = 21
        #     num_values_W = 21
        #     num_values_filter = 5
        #     num_filters = 3
        #     # Input signal: 5 data points, 3 channels, 10 values.
        #     x = np.random.rand(num_data_points, num_channels, num_values_data)
        #     # Filters: 3 filters, 3 channels, 4 values.
        #     y = np.random.rand(num_filters, num_channels, num_values_filter)
        #     # Bias: one for each filter
        #     b = np.random.rand(num_filters)
        #     # get the expected result
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, _ = conv_forward_naive_2D(x, y, b, conv_param)
        #     self.logger.debug("expected result: " + str(expected_result))
        #
        #     conv = PyTorchConv1dFunction()
        #     result = conv.forward(ctx=None, input=torch.from_numpy(x),
        #                           filter=torch.from_numpy(y),
        #                           bias=torch.from_numpy(b))
        #     self.logger.debug("obtained result: " + str(result))
        #     np.testing.assert_array_almost_equal(
        #         result, np.array(expected_result))
        #
        # def test_FunctionBackwardNoCompressionWithBias(self):
        #     x = np.array([[[1.0, 2.0, 3.0]]])
        #     y = np.array([[[2.0, 1.0]]])
        #     b = np.array([2.0])
        #     dtype = torch.float
        #     x_torch = tensor(x, requires_grad=True, dtype=dtype)
        #     y_torch = tensor(y, requires_grad=True, dtype=dtype)
        #     b_torch = tensor(b, requires_grad=True, dtype=dtype)
        #
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, cache = conv_forward_naive_1D(x, y, b,
        #                                                    conv_param)
        #     ctx = MockContext()
        #     ctx.set_needs_input_grad(3)
        #     result_torch = PyTorchConv1dFunction.forward(ctx, x_torch,
        #                                                  y_torch, b_torch)
        #     result = result_torch.detach().numpy()
        #     np.testing.assert_array_almost_equal(
        #         result, np.array(expected_result))
        #
        #     dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        #     # get the expected result from the backward pass
        #     expected_dx, expected_dw, expected_db = \
        #         conv_backward_naive_1D(dout.numpy(), cache)
        #
        #     dx, dw, db, _, _, _, _, _ = PyTorchConv1dFunction.backward(ctx, dout)
        #
        #     self.logger.debug("expected dx: " + str(expected_dx))
        #     self.logger.debug("computed dx: " + str(dx))
        #
        #     # are the gradients correct
        #     np.testing.assert_array_almost_equal(dx.detach().numpy(),
        #                                          expected_dx)
        #     np.testing.assert_array_almost_equal(dw.detach().numpy(),
        #                                          expected_dw)
        #     np.testing.assert_array_almost_equal(db.detach().numpy(),
        #                                          expected_db)
        #
        # def test_FunctionBackwardNoCompressionNoBias(self):
        #     x = np.array([[[1.0, 2.0, 3.0]]])
        #     y = np.array([[[2.0, 1.0]]])
        #     b = np.array([0.0])
        #     dtype = torch.float
        #     x_torch = tensor(x, requires_grad=True, dtype=dtype)
        #     y_torch = tensor(y, requires_grad=True, dtype=dtype)
        #     b_torch = tensor(b, requires_grad=True, dtype=dtype)
        #
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, cache = conv_forward_naive_1D(x, y, b,
        #                                                    conv_param)
        #
        #     result_torch = PyTorchConv1dFunction.apply(x_torch, y_torch,
        #                                                b_torch)
        #     result = result_torch.detach().numpy()
        #     np.testing.assert_array_almost_equal(
        #         result, np.array(expected_result))
        #
        #     dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        #     # get the expected result from the backward pass
        #     expected_dx, expected_dw, expected_db = \
        #         conv_backward_naive_1D(dout.numpy(), cache)
        #
        #     result_torch.backward(dout)
        #
        #     # are the gradients correct
        #     np.testing.assert_array_almost_equal(x_torch.grad,
        #                                          expected_dx)
        #     np.testing.assert_array_almost_equal(y_torch.grad,
        #                                          expected_dw)
        #     np.testing.assert_array_almost_equal(b_torch.grad,
        #                                          expected_db)
        #
        # def test_FunctionBackwardWithPooling(self):
        #     x = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -2.0]]])
        #     y = np.array([[[2.0, 1.0, 3.0, 1.0, -3.0]]])
        #     b = np.array([0.0])
        #     dtype = torch.float
        #     x_torch = tensor(x, requires_grad=True, dtype=dtype)
        #     y_torch = tensor(y, requires_grad=True, dtype=dtype)
        #     b_torch = tensor(b, requires_grad=True, dtype=dtype)
        #
        #     conv_param = {'pad': 0, 'stride': 1}
        #     accurate_expected_result, cache = conv_forward_naive_1D(x, y, b,
        #                                                             conv_param)
        #     print("Accurate expected result: ", accurate_expected_result)
        #
        #     approximate_expected_result = np.array(
        #         [[[-2.105834, 0.457627, 8.501472, 20.74531]]])
        #     print("Approximate expected result: ", approximate_expected_result)
        #
        #     out_size = approximate_expected_result.shape[-1]
        #
        #     result_torch = PyTorchConv1dFunction.apply(x_torch, y_torch, b_torch,
        #                                                None, None, out_size)
        #     result = result_torch.detach().numpy()
        #     np.testing.assert_array_almost_equal(
        #         x=np.array(approximate_expected_result), y=result,
        #         err_msg="Expected x is different from computed y.")
        #
        #     self._check_delta(actual_result=result,
        #                       accurate_expected_result=accurate_expected_result,
        #                       delta=6.8)
        #
        #     dout = tensor([[[0.1, -0.2, 0.3, -0.1]]], dtype=dtype)
        #     # get the expected result from the backward pass
        #     expected_dx, expected_dw, expected_db = \
        #         conv_backward_naive_1D(dout.numpy(), cache)
        #
        #     result_torch.backward(dout)
        #
        #     approximate_expected_dx = np.array(
        #         [[[0.052956, 0.120672, 0.161284, 0.150332, 0.089258,
        #            0.005318, -0.063087, -0.087266, -0.063311, -0.012829]]])
        #
        #     # are the gradients correct
        #     np.testing.assert_array_almost_equal(
        #         x=approximate_expected_dx, y=x_torch.grad,
        #         err_msg="Expected x is different from computed y.")
        #
        #     self._check_delta(actual_result=x_torch.grad,
        #                       accurate_expected_result=expected_dx, delta=0.95)
        #
        #     approximate_expected_dw = np.array(
        #         [[[0.129913, 0.249468, 0.429712, 0.620098, 0.748242]]])
        #     np.testing.assert_array_almost_equal(
        #         x=approximate_expected_dw, y=y_torch.grad,
        #         err_msg="Expected x is different from computed y.")
        #
        #     self._check_delta(actual_result=y_torch.grad,
        #                       accurate_expected_result=expected_dw, delta=0.2)
        #
        #     np.testing.assert_array_almost_equal(b_torch.grad,
        #                                          expected_db)
        #
        # def _check_delta(self, actual_result, accurate_expected_result, delta):
        #     """
        #     Compare if that the difference between the two objects is more than the
        #     given delta.
        #
        #     :param actual_result: the computed result
        #     :param accurate_expected_result: the expected accurate result
        #     :param delta: compare if that the difference between the two objects
        #     is more than the given delta
        #     """
        #     print("actual_result: {}".format(actual_result))
        #     print("accurate_expected_result: {}".format(accurate_expected_result))
        #     result_flat = actual_result[0][0]
        #     accurate_expected_flat = accurate_expected_result[0][0]
        #     for index, item in enumerate(result_flat):
        #         self.assertAlmostEqual(
        #             first=accurate_expected_flat[index], second=item, delta=delta,
        #             msg="The approximate result is not within delta={} of the "
        #                 "accurate result!".format(delta))
        #
        # def test_FunctionBackwardCompressionBias(self):
        #     x = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 10.0]]])
        #     y = np.array([[[2.0, 1.0, -3.0]]])
        #     b = np.array([1.0])
        #     dtype = torch.float
        #     x_torch = tensor(x, requires_grad=True, dtype=dtype)
        #     y_torch = tensor(y, requires_grad=True, dtype=dtype)
        #     b_torch = tensor(b, requires_grad=True, dtype=dtype)
        #
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)
        #     print("expected result: ", expected_result)
        #
        #     # 1 index back
        #     result_torch = PyTorchConv1dFunction.apply(x_torch, y_torch, b_torch,
        #                                                None, 1)
        #     result = result_torch.detach().numpy()
        #     compressed_expected_result = np.array(
        #         [[[-2.25, -5.749999, -2.25, 15.249998, -18.25]]])
        #     np.testing.assert_array_almost_equal(
        #         x=compressed_expected_result, y=result,
        #         err_msg="Expected x is different from computed y.")
        #
        #     dout = tensor([[[0.1, -0.2, -0.3, 0.3, 0.1]]], dtype=dtype)
        #     # get the expected result from the backward pass
        #     expected_dx, expected_dw, expected_db = \
        #         conv_backward_naive_1D(dout.numpy(), cache)
        #
        #     result_torch.backward(dout)
        #
        #     # are the gradients correct
        #     print("accurate expected_dx: ", expected_dx)
        #     approximate_dx = np.array(
        #         [[[0.175, -0.275, -1.125, 0.925, 1.375, -0.775, -0.325]]])
        #     np.testing.assert_array_almost_equal(
        #         x=approximate_dx, y=x_torch.grad,
        #         err_msg="Expected approximate x is different from computed y. The "
        #                 "exact x (that represents dx) is: {}".format(expected_dx))
        #     print("accurate expected_dw: ", expected_dw)
        #     approximate_dw = np.array([[[0.675, -0.375, -1.125]]])
        #     np.testing.assert_array_almost_equal(
        #         x=approximate_dw, y=y_torch.grad,
        #         err_msg="Expected approximate x is different from computed y. The "
        #                 "exact x (that represents dw) is: {}".format(expected_dw))
        #     np.testing.assert_array_almost_equal(
        #         x=expected_db, y=b_torch.grad,
        #         err_msg="Expected approximate x is different from computed y.")
        #
        # def test_FunctionBackwardNoCompression2Channels(self):
        #     x = np.array([[[1.0, 2.0, 3.0], [-1.0, -3.0, 2.0]]])
        #     y = np.array([[[2.0, 1.0], [-2.0, 3.0]]])
        #     # still it is only a single filter but with 2 channels
        #     b = np.array([0.0])
        #     dtype = torch.float
        #     x_torch = tensor(x, requires_grad=True, dtype=dtype)
        #     y_torch = tensor(y, requires_grad=True, dtype=dtype)
        #     b_torch = tensor(b, requires_grad=True, dtype=dtype)
        #
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)
        #
        #     result_torch = PyTorchConv1dFunction.apply(x_torch, y_torch, b_torch)
        #     result = result_torch.detach().numpy()
        #     np.testing.assert_array_almost_equal(
        #         result, np.array(expected_result))
        #
        #     dout = tensor([[[0.1, -0.2]]], dtype=dtype)
        #     # get the expected result from the backward pass
        #     expected_dx, expected_dw, expected_db = \
        #         conv_backward_naive_1D(dout.numpy(), cache)
        #
        #     result_torch.backward(dout)
        #
        #     print()
        #     print("expected dx: " + str(expected_dx))
        #     print("computed dx: {}".format(x_torch.grad))
        #
        #     print("expected dw: {}".format(expected_dw))
        #     print("computed dw: {}".format(y_torch.grad))
        #     #
        #     # self.logger.debug("expected db: ", expected_db)
        #     # self.logger.debug("computed db: ", b_torch.grad)
        #
        #     # are the gradients correct
        #     np.testing.assert_array_almost_equal(
        #         x=expected_dx, y=x_torch.grad,
        #         err_msg="Expected x is different from computed y.")
        #     np.testing.assert_array_almost_equal(y_torch.grad,
        #                                          expected_dw)
        #     np.testing.assert_array_almost_equal(b_torch.grad,
        #                                          expected_db)
        #
        # def test_FunctionForwardWithCompression(self):
        #     # test with compression
        #     x = np.array([[[1., 2., 3.]]])
        #     y = np.array([[[2., 1.]]])
        #     b = np.array([0.0])
        #     expected_result = [3.5, 7.5]
        #     conv = PyTorchConv1dFunction()
        #     result = conv.forward(
        #         ctx=None, input=torch.from_numpy(x), filter=torch.from_numpy(y),
        #         bias=torch.from_numpy(b), index_back=1)
        #     np.testing.assert_array_almost_equal(
        #         result, np.array([[expected_result]]))
        #
        # def test_AutogradForwardNoCompression(self):
        #     x = np.array([[[1., 2., 3.]]])
        #     y = np.array([[[2., 1.]]])
        #     b = np.array([0.0])
        #     # get the expected results from numpy correlate
        #     expected_result = np.correlate(x[0, 0, :], y[0, 0, :],
        #                                    mode="valid")
        #     conv = PyTorchConv2dAutograd(filter=torch.from_numpy(y),
        #                                  bias=torch.from_numpy(b))
        #     result = conv.forward(input=torch.from_numpy(x))
        #
        #     np.testing.assert_array_almost_equal(
        #         result, np.array([[expected_result]]))
        #
        # def test_AutogradForwardWithCompression(self):
        #     # test with compression
        #     x = np.array([[[1., 2., 3.]]])
        #     y = np.array([[[2., 1.]]])
        #     b = np.array([0.0])
        #     expected_result = [3.5, 7.5]
        #     conv = PyTorchConv1dAutograd(
        #         filter=torch.from_numpy(y), bias=torch.from_numpy(b), index_back=1)
        #     result = conv.forward(input=torch.from_numpy(x))
        #     np.testing.assert_array_almost_equal(
        #         result, np.array([[expected_result]]))
        #
        # def test_FunctionForwardBackwardRandom(self):
        #     num_channels = 3
        #     num_data_points = 11
        #     num_values_data = 21
        #     num_values_filter = 5
        #     num_filters = 3
        #     # Input signal: 5 data points, 3 channels, 10 values.
        #     x = np.random.rand(num_data_points, num_channels, num_values_data)
        #     # Filters: 3 filters, 3 channels, 4 values.
        #     y = np.random.rand(num_filters, num_channels, num_values_filter)
        #     # Bias: one for each filter
        #     b = np.random.rand(num_filters)
        #     # get the expected result
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, _ = conv_forward_naive_1D(x, y, b, conv_param)
        #     self.logger.debug("expected result: " + str(expected_result))
        #
        #     dtype = torch.float
        #     x_torch = tensor(x, requires_grad=True, dtype=dtype)
        #     y_torch = tensor(y, requires_grad=True, dtype=dtype)
        #     b_torch = tensor(b, requires_grad=True, dtype=dtype)
        #     conv = PyTorchConv1dFunction()
        #     result_torch = conv.forward(ctx=None, input=x_torch, filter=y_torch,
        #                                 bias=b_torch)
        #     result = result_torch.detach().numpy()
        #     self.logger.debug("obtained result: " + str(result))
        #     np.testing.assert_array_almost_equal(result, np.array(expected_result))
        #
        #     conv_param = {'pad': 0, 'stride': 1}
        #     expected_result, cache = conv_forward_naive_1D(x, y, b, conv_param)
        #
        #     # dout = tensor(result/100.0, dtype=dtype)
        #     dout = torch.randn(result_torch.shape)
        #     # get the expected result from the backward pass
        #     expected_dx, expected_dw, expected_db = \
        #         conv_backward_naive_1D(dout.numpy(), cache)
        #
        #     result_torch.backward(dout)
        #
        #     # print()
        #     # print("expected dx: " + str(expected_dx))
        #     # print("computed dx: {}".format(x_torch.grad))
        #     #
        #     # print("expected dw: {}".format(expected_dw))
        #     # print("computed dw: {}".format(y_torch.grad))
        #     # self.logger.debug("expected db: ", expected_db)
        #     # self.logger.debug("computed db: ", b_torch.grad)
        #
        #     # are the gradients correct
        #     np.testing.assert_array_almost_equal(
        #         x=expected_dx, y=x_torch.grad,
        #         err_msg="Expected x is different from computed y.")
        #     np.testing.assert_array_almost_equal(
        #         x=expected_dw, y=y_torch.grad, decimal=4,
        #         err_msg="Expected x is different from computed y.")
        #     np.testing.assert_array_almost_equal(
        #         x=expected_db, y=b_torch.grad, decimal=5,
        #         err_msg="Expected x is different from computed y.")

        if __name__ == '__main__':
            unittest.main()
