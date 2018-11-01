import logging
import unittest

import numpy as np
import torch
from torch import tensor

from cnns.nnlib.layers import conv_forward_naive, conv_backward_naive
from cnns.nnlib.pytorch_layers.conv2D_fft \
    import Conv2dfftAutograd, Conv2dfftFunction, Conv2dfft
from cnns.nnlib.pytorch_layers.pytorch_utils import MockContext
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

    def testAutogradForwardNoCompression(self):
        print("Don't use next power of 2.")
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])

        convManual = Conv2dfft(filter_value=y, bias=b, index_back=0,
                               use_next_power2=False)
        resultManual = convManual.forward(input=x)
        print("result of manual convolution: ", resultManual)

        convAuto = Conv2dfftAutograd(filter_value=y, bias=b, index_back=0,
                                     use_next_power2=False)
        resultAuto = convAuto.forward(input=x)

        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])

        np.testing.assert_array_almost_equal(
            x=expect, y=resultManual,
            err_msg="The expected array x and computed manually y are not "
                    "almost equal.")

        np.testing.assert_array_almost_equal(
            x=expect, y=resultAuto,
            err_msg="The expected array x and computed auto y are not almost "
                    "equal.")

    def test_ForwardNoCompressionForConv2dfft(self):
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        convManual = Conv2dfft(filter_value=y, bias=b, index_back=0,
                               use_next_power2=True)
        result = convManual.forward(input=x)
        print("result of manual convolution: ", result)
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
        conv = Conv2dfftFunction()

        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              index_back=None, use_next_power2=False)
        print("Result of conv function: ", result)

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
        conv = Conv2dfftFunction()
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
        conv = Conv2dfftFunction()
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
        conv = Conv2dfftFunction()
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
        conv = Conv2dfftFunction()
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
        conv = Conv2dfftFunction()
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

    def test_FunctionForwardRandom(self):
        num_channels = 3
        num_data_points = 11
        input_H = 21
        input_W = 21
        filter_H = 5
        filter_W = 5
        num_filters = 3
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, input_H,
                           input_W)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, filter_H, filter_W)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive(x, y, b, conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardRandom(self):
        num_channels = 3
        num_data_points = 11
        input_H = 21
        input_W = 21
        filter_H = 5
        filter_W = 5
        num_filters = 3
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, input_H,
                           input_W)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, filter_H, filter_W)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive(x, y, b, conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        conv = Conv2dfftAutograd(filter_value=torch.from_numpy(y),
                                 bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result, np.array(expected_result))

    def test_FunctionForwardSpectralPooling(self):
        x = np.array([[[[1., 2., 3., 4., 5.],
                        [6., 7., 8., 1., 2.],
                        [2., 3., 1., 0., 1.],
                        [1., 2., 3., -1., -2.],
                        [0., 1., 3., 1., 2.]
                        ]]])
        y = np.array([[[[2., 1.], [-1.0, 2.0]]]])
        b = np.array([0.0])

        # Full result.
        conv_param = {'pad': 0, 'stride': 1}
        full_expected_result, _ = conv_forward_naive(x=x, w=y, b=b,
                                                     conv_param=conv_param)
        print()
        print("full expected result: ", full_expected_result)

        # get the expected results from numpy correlate
        # expected_result = np.array([[[[10.103396, 12.630585, 11.697527],
        #                               [12.558281, 13.923859, 11.561422],
        #                               [11.473415, 11.409614, 8.187342]]]])
        expected_result = np.array([[[[11.2787, 14.2694, 12.6907],
                                      [14.0552, 15.6585, 12.3298],
                                      [12.0275, 11.8809, 7.7573]]]])
        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b), out_size=3)
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result, decimal=4,
            err_msg="Expected x is different from computed y.")

    def test_FunctionBackwardNoCompressionWithBias(self):
        x = np.array([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 1.0],
                        [1.0, -1.0, -2.0]]]])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]]]])
        b = np.array([2.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        ctx = MockContext()
        ctx.set_needs_input_grad(3)
        result_torch = Conv2dfftFunction.forward(
            ctx, input=x_torch, filter=y_torch, bias=b_torch)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        dout = tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.numpy(), cache)

        dx, dw, db, _, _, _, _, _, _, _, _, _, _ = Conv2dfftFunction.backward(
            ctx, dout)

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
        x = np.array([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 1.0],
                        [1.0, -1.0, -2.0]]]])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]]]])
        b = np.array([0.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("\nexpected result: ", expected_result)

        ctx = MockContext()
        ctx.set_needs_input_grad(3)
        result_torch = Conv2dfftFunction.forward(
            ctx, input=x_torch, filter=y_torch, bias=b_torch)
        result = result_torch.detach().numpy()
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        result_torch_2 = Conv2dfft(filter_value=y_torch, bias=b_torch).forward(
            input=x_torch)
        result2 = result_torch_2.detach().numpy()
        print("actual result 2: ", result2)
        np.testing.assert_array_almost_equal(result2, np.array(expected_result))

        dout = tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.numpy(), cache)

        dx, dw, db, _, _, _, _, _, _, _, _, _, _ = Conv2dfftFunction.backward(
            ctx, dout)

        self.logger.debug("expected dx: " + str(expected_dx))
        self.logger.debug("computed dx: " + str(dx))

        # are the gradients correct
        np.testing.assert_array_almost_equal(dx.detach().numpy(),
                                             expected_dx)
        np.testing.assert_array_almost_equal(dw.detach().numpy(),
                                             expected_dw)
        np.testing.assert_array_almost_equal(db.detach().numpy(),
                                             expected_db)

    def test_FunctionBackwardNoCompressionConv2dfft(self):
        x = np.array([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 1.0],
                        [1.0, -1.0, -2.0]]]])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]]]])
        b = np.array([1.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        conv = Conv2dfft(filter_value=y_torch, bias_value=b_torch, index_back=0)

        result_torch = conv.forward(input=x_torch)
        result = result_torch.detach().numpy()
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        dout = tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.numpy(), cache)

        result_torch.backward(dout)
        assert conv.is_manual[0] == 1

        self.logger.debug("expected dx: " + str(expected_dx))
        self.logger.debug("computed dx: " + str(x_torch.grad))

        # are the gradients correct
        np.testing.assert_array_almost_equal(x_torch.grad.detach().numpy(),
                                             expected_dx)
        np.testing.assert_array_almost_equal(y_torch.grad.detach().numpy(),
                                             expected_dw)
        np.testing.assert_array_almost_equal(b_torch.grad.detach().numpy(),
                                             expected_db)

    def _check_delta2D(self, actual_result, accurate_expected_result, delta):
        """
        Compare if the difference between the two objects is more than the
        given delta.

        :param actual_result: the computed result
        :param accurate_expected_result: the expected accurate result
        :param delta: compare if that the difference between the two objects
        is more than the given delta
        """
        print("actual_result: {}".format(actual_result))
        print("accurate_expected_result: {}".format(accurate_expected_result))
        result_flat = actual_result[0][0]
        accurate_expected_flat = accurate_expected_result[0][0]
        for index_h, item_h in enumerate(result_flat):
            for index_w, item_w in enumerate(item_h):
                self.assertAlmostEqual(
                    first=accurate_expected_flat[index_h][index_w],
                    second=index_w, delta=delta,
                    msg="The approximate result is not within delta={} of the "
                        "accurate result!".format(delta))

    def test_FunctionBackwardWithPooling(self):
        x = np.array([[[[1., 2., 3., 4., 5.],
                        [6., 7., 8., 1., 2.],
                        [2., 3., 1., 0., 1.],
                        [1., 2., 3., -1., -2.],
                        [0., 1., 3., 1., 2.]
                        ]]])
        y = np.array([[[[2., 1.], [-1.0, 2.0]]]])
        b = np.array([0.0])

        # Full result.
        conv_param = {'pad': 0, 'stride': 1}
        full_expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                         conv_param=conv_param)
        print()
        print("full expected result: ", full_expected_result)

        # get the expected results from numpy correlate
        # expected_result = np.array([[[[10.103396, 12.630585, 11.697527],
        #                               [12.558281, 13.923859, 11.561422],
        #                               [11.473415, 11.409614, 8.187342]]]])
        expected_result = np.array([[[[11.2787, 14.2694, 12.6907],
                                      [14.0552, 15.6585, 12.3298],
                                      [12.0275, 11.8809, 7.7573]]]])
        conv = Conv2dfftFunction()

        x_torch = torch.tensor(data=x, requires_grad=True)
        y_torch = torch.tensor(data=y, requires_grad=True)
        b_torch = torch.tensor(data=b, requires_grad=True)

        out_size = 3

        result_torch = conv.forward(ctx=None, input=x_torch, filter=y_torch,
                                    bias=b_torch, out_size=out_size)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result, decimal=4,
            err_msg="Expected x is different from computed y.")

        # Prevent any interference/overlap of variables in manual and auto
        # differentiation.
        x_torch_auto = torch.tensor(data=x, requires_grad=True)
        y_torch_auto = torch.tensor(data=y, requires_grad=True)
        b_torch_auto = torch.tensor(data=b, requires_grad=True)

        convAuto = Conv2dfftAutograd(filter_value=y_torch_auto,
                                     bias=b_torch_auto, out_size=out_size)
        resultAuto = convAuto.forward(input=x_torch_auto)

        dout_np = np.array([[[[0.1, -0.2, 0.3],
                              [-0.1, 0.1, 0.2],
                              [-0.2, 1.1, -1.2]]]])
        dout = tensor(dout_np)

        resultAuto.backward(dout)
        print("x auto grad: ", x_torch_auto.grad)
        print("y auto grad: ", y_torch_auto.grad)
        print("b auto grad: ", b_torch_auto.grad)

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.numpy(), cache)

        result_torch.backward(dout)

        # approximate_expected_dx = np.array(
        #     [[[[0.0306, 0.1016, 0.1293, 0.0976, 0.0249],
        #        [0.0815, 0.1438, 0.1321, 0.0534, -0.0463],
        #        [0.1171, 0.1399, 0.0813, -0.0245, -0.1154],
        #        [0.1164, 0.0923, 0.0066, -0.0904, -0.1420],
        #        [0.0799, 0.0287, -0.0482, -0.1058, -0.1104]]]])

        approximate_expected_dx = np.array(
            [[[[0.0004, 0.1056, 0.1608, 0.1246, 0.0241],
               [0.0604, 0.1825, 0.1858, 0.0676, -0.0829],
               [0.1250, 0.1951, 0.1164, -0.0518, -0.1829],
               [0.1456, 0.1338, 0.0051, -0.1437, -0.2005],
               [0.1066, 0.0448, -0.0645, -0.1389, -0.1225]]]])

        print("manual torch grad: ", x_torch.grad)

        # Are the gradients correct?
        np.testing.assert_array_almost_equal(
            x=approximate_expected_dx, y=x_torch.grad, decimal=4,
            err_msg="Expected x is different from computed y.")

        self._check_delta2D(actual_result=x_torch.grad,
                            accurate_expected_result=expected_dx, delta=5.4)

        print("Expected fully correct dw: ", expected_dw)
        print("actual result for dw from y_torch.grad: ", y_torch.grad)

        # approximate_expected_dw = np.array([[[[0.844089, 1.41447],
        #                                       [1.221608, 1.32085]]]])

        approximate_expected_dw = np.array([[[[1.1816, 1.8317],
                                              [1.5589, 1.4568]]]])

        np.testing.assert_array_almost_equal(
            x=approximate_expected_dw, y=y_torch.grad, decimal=4,
            err_msg="Expected x is different from computed y.")

        self._check_delta2D(actual_result=y_torch.grad,
                            accurate_expected_result=expected_dw, delta=4.0)

        np.testing.assert_array_almost_equal(b_torch.grad,
                                             expected_db)

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
    #     result_torch = Conv1dfftFunction.apply(x_torch, y_torch, b_torch,
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
    #     result_torch = Conv1dfftFunction.apply(x_torch, y_torch, b_torch,
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
    #     result_torch = Conv1dfftFunction.apply(x_torch, y_torch, b_torch)
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
    #     conv = Conv1dfftFunction()
    #     result = conv.forward(
    #         ctx=None, input=torch.from_numpy(x), filter=torch.from_numpy(y),
    #         bias=torch.from_numpy(b), index_back=1)
    #     np.testing.assert_array_almost_equal(
    #         result, np.array([[expected_result]]))

    def test_AutogradForwardNoCompression(self):
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        conv = Conv2dfftAutograd(filter_value=y, bias=b, index_back=0)
        result = conv.forward(input=x)
        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal.")

    def test_AutogradForwardWithCompression(self):
        # A single input 2D map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        conv = Conv2dfftAutograd(filter_value=y, bias=b, index_back=1,
                                 use_next_power2=False)
        result = conv.forward(input=x)
        expect = np.array([[[[21.5, 22.0], [17.5, 13.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal")

    def test_FunctionForwardBackwardRandom(self):
        num_channels = 3
        num_data_points = 11
        input_H = 21
        input_W = 21
        filter_H = 5
        filter_W = 5
        num_filters = 3
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, input_H, input_W)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, filter_H, filter_W)
        # Bias: one for each filter
        b = np.random.rand(num_filters)
        # get the expected result
        conv_param = {'pad': 0, 'stride': 1}
        expected_result, _ = conv_forward_naive(x=x, w=y, b=b,
                                                conv_param=conv_param)
        self.logger.debug("expected result: " + str(expected_result))

        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)
        conv = Conv2dfftFunction()
        result_torch = conv.forward(ctx=None, input=x_torch, filter=y_torch,
                                    bias=b_torch)
        result = result_torch.detach().numpy()
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result, decimal=4,
            err_msg="The expected result x differs from the computed result y.")

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x, y, b, conv_param)

        # dout = tensor(result/100.0, dtype=dtype)
        dout = torch.randn(result_torch.shape)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.numpy(), cache)

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
            x=expected_dx, y=x_torch.grad, decimal=5,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_dw, y=y_torch.grad, decimal=4,
            err_msg="Expected x is different from computed y.")
        np.testing.assert_array_almost_equal(
            x=expected_db, y=b_torch.grad, decimal=4,
            err_msg="Expected x is different from computed y.")


if __name__ == '__main__':
    unittest.main()
