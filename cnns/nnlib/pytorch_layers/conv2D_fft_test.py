import logging
import unittest
import time
import numpy as np
import torch
from torch import tensor
from torch.nn import functional as F
from cnns.nnlib.layers import conv_forward_naive, conv_backward_naive
from cnns.nnlib.pytorch_layers.conv2D_fft \
    import Conv2dfftAutograd, Conv2dfftFunction, Conv2dfft
from cnns.nnlib.pytorch_layers.pytorch_utils import MockContext
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.pytorch_layers.test_data.cifar10_image import cifar10_image
from cnns.nnlib.pytorch_layers.test_data.cifar10_lenet_filter import \
    cifar10_lenet_filter
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.arguments import Arguments


class TestPyTorchConv2d(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv2D_reuse_map_fft.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def testAutogradForwardNoCompression(self):
        print("\nDon't use next power of 2.")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is not available")
        dtype = torch.float

        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=device, dtype=dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]], device=device, dtype=dtype)
        b = tensor([0.0], device=device, dtype=dtype)

        convManual = Conv2dfft(weight_value=y, bias_value=b)
        resultManual = convManual.forward(input=x)
        print("result of manual convolution: ", resultManual)

        convAuto = Conv2dfftAutograd(weight_value=y, bias_value=b)
        resultAuto = convAuto.forward(input=x)

        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])

        np.testing.assert_array_almost_equal(
            x=expect, y=resultManual.cpu().detach().numpy(), decimal=5,
            err_msg="The expected array x and computed manually y are not "
                    "almost equal.")

        np.testing.assert_array_almost_equal(
            x=expect, y=resultAuto.cpu().detach().numpy(), decimal=5,
            err_msg="The expected array x and computed auto y are not almost "
                    "equal.")

    def test_ForwardNoCompressionForConv2dfft(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is not available")
        dtype = torch.float
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=device, dtype=dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]], device=device, dtype=dtype)
        b = tensor([0.0], device=device, dtype=dtype)
        convManual = Conv2dfft(weight_value=y, bias_value=b)
        result = convManual.forward(input=x)
        print("result of manual convolution: ", result)
        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result.detach().numpy(),
            err_msg="The expected array x and computed y are not almost equal.")

    def test_FunctionForwardNoCompression(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is not available")
        dtype = torch.float
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=device, dtype=dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]], device=device, dtype=dtype)
        b = tensor([0.0], device=device, dtype=dtype)
        conv = Conv2dfftFunction()

        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              args=Arguments())
        print("Result of conv function: ", result)

        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal.")

    def test_2_channels_2_filters(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is not available")
        dtype = torch.float
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]],
                     [[1., 1., 2.], [2., 3., 1.], [2., -1., 3.]]]],
                   device=device, dtype=dtype)
        print("shape of x: ", x.size())
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]], [[-1.0, 2.0], [3.0, -2.0]]],
                    [[[-1.0, 1.0], [2.0, 3.0]], [[-2.0, 1.0], [1.0, -3.0]]]],
                   device=device, dtype=dtype)
        b = tensor([0.0, 0.0], device=device, dtype=dtype)
        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              args=Arguments())
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
        result = conv.forward(ctx=None, input=x, filter=y, bias=b)
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
        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              args=Arguments(index_back=1, preserve_energy=100))
        # expect = np.array([[[[21.5, 22.0], [17.5, 13.]]]])
        expect = np.array([[[[21.75, 21.75], [18.75, 13.75]]]])
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
        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              args=Arguments())
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
        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              padding=(1, 1), stride=(2, 2),
                              args=Arguments(stride_type=StrideType.STANDARD))
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

        conv = Conv2dfft(weight_value=torch.from_numpy(y),
                         bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            result.detach().numpy(), np.array(expected_result))

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
        # expected_result = np.array([[[[11.2787, 14.2694, 12.6907],
        #                               [14.0552, 15.6585, 12.3298],
        #                               [12.0275, 11.8809, 7.7573]]]])
        expected_result = np.array([[[[12.2992, 13.6678, 10.92],
                                      [15.9293, 16.679, 11.7282],
                                      [13.3441, 13.755, 8.7778]]]])
        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=torch.from_numpy(x),
                              filter=torch.from_numpy(y),
                              bias=torch.from_numpy(b),
                              out_size=3)
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
        is_manual = tensor([0])
        result_torch = Conv2dfftFunction.forward(
            ctx, input=x_torch, filter=y_torch, bias=b_torch, args=Arguments(),
            is_manual=is_manual)
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        dout = tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=dtype)
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.numpy(), cache)

        dx, dw, db, _, _, _, _, _, _ = Conv2dfftFunction.backward(
            ctx, dout)

        self.logger.debug("\nexpected dx: " + str(expected_dx))
        self.logger.debug("\ncomputed dx: " + str(dx))
        assert is_manual[0] == 1

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

        is_manual = tensor([0])
        result_torch = Conv2dfftFunction.forward(
            ctx, input=x_torch, filter=y_torch, bias=b_torch,
            is_manual=is_manual, args=Arguments())
        result = result_torch.detach().numpy()
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        result_torch_2 = Conv2dfft(weight_value=y_torch, bias=b_torch).forward(
            input=x_torch)
        result2 = result_torch_2.detach().numpy()
        print("actual result 2: ", result2)
        np.testing.assert_array_almost_equal(result2, np.array(expected_result))

        dout = torch.autograd.Variable(
            tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=dtype))
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.numpy(), cache)

        dx, dw, db, _, _, _, _, _, _ = Conv2dfftFunction.backward(ctx, dout)

        assert is_manual[0] == 1

        self.logger.debug("\nexpected dx: " + str(expected_dx))
        self.logger.debug("\ncomputed dx: " + str(dx))

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
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is not available")
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype, device=device)
        y_torch = tensor(y, requires_grad=True, dtype=dtype, device=device)
        b_torch = tensor(b, requires_grad=True, dtype=dtype, device=device)

        conv = Conv2dfft(weight_value=y_torch, bias_value=b_torch)
        # conv = Conv2dfft()

        result_torch = conv.forward(input=x_torch)
        dout = torch.autograd.Variable(
            tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=dtype, device=device))
        result_torch.backward(dout)
        assert conv.is_manual[0] == 1

        result_torch = result_torch.cpu().detach()
        print("result torch: ", result_torch)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.cpu().numpy(), cache)

        result = result_torch.detach()
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        # Are the gradients correct?

        self.logger.debug("expected dx: " + str(expected_dx))
        self.logger.debug("computed dx: " + str(x_torch.grad.cpu().numpy()))

        print("expected dx: " + str(expected_dx))
        print("computed dx: " + str(x_torch.grad.cpu().numpy()))

        np.testing.assert_array_almost_equal(
            x_torch.grad.cpu().detach().numpy(), expected_dx)

        self.logger.debug("expected dw: " + str(expected_dw))
        self.logger.debug("computed dw: " + str(y_torch.grad.cpu().numpy()))

        print("expected dw: " + str(expected_dw))
        print("computed dw: " + str(y_torch.grad.cpu().numpy()))

        np.testing.assert_array_almost_equal(
            y_torch.grad.cpu().detach().numpy(), expected_dw)
        np.testing.assert_array_almost_equal(
            b_torch.grad.cpu().detach().numpy(), expected_db)

    def test_FunctionBackwardNoCompressionConv2dfft2channels(self):
        x = np.array([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 1.0],
                        [1.0, -1.0, -2.0]],
                       [[2.0, 3.0, 3.0],
                        [4.0, 1.0, 1.0],
                        [-1.0, -1.0, -3.0]]
                       ]])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]], [[1.0, 0.0], [2.0, 1.0]]]])
        b = np.array([1.0])
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is not available")
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype, device=device)
        print("x size: ", x_torch.size())
        y_torch = tensor(y, requires_grad=True, dtype=dtype, device=device)
        print("y size: ", y_torch.size())
        b_torch = tensor(b, requires_grad=True, dtype=dtype, device=device)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        conv = Conv2dfft(weight_value=y_torch, bias_value=b_torch)

        result_torch = conv.forward(input=x_torch)
        # dout = torch.autograd.Variable(
        #     tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=dtype))
        dout = tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=dtype,
                      device=device)
        result_torch.backward(dout)
        assert conv.is_manual[0] == 1

        result = result_torch.cpu().detach().numpy()
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.cpu().numpy(), cache)

        self.logger.debug("expected dx: " + str(expected_dx))
        self.logger.debug("computed dx: " + str(x_torch.grad.cpu().numpy()))

        print("expected dx: " + str(expected_dx))
        print("computed dx: " + str(x_torch.grad.cpu().numpy()))

        # are the gradients correct

        np.testing.assert_array_almost_equal(
            x_torch.grad.cpu().detach().numpy(), expected_dx, decimal=5)

        np.testing.assert_array_almost_equal(
            y_torch.grad.cpu().detach().numpy(), expected_dw, decimal=5)

        np.testing.assert_array_almost_equal(
            b_torch.grad.cpu().detach().numpy(), expected_db, decimal=5)

    def test_FunctionBackwardNoCompressionConv2dfft1Image(self):
        x = np.array([[
            [[2.0, 1.0, 3.0],
             [4.0, 2.0, 1.0],
             [-1.0, -2.0, -3.0]]
        ]])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]]]])
        b = np.array([1.0])
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is not available")
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype, device=device)
        y_torch = tensor(y, requires_grad=True, dtype=dtype, device=device)
        b_torch = tensor(b, requires_grad=True, dtype=dtype, device=device)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        conv = Conv2dfft(weight_value=y_torch, bias_value=b_torch)

        result_torch = conv.forward(input=x_torch)
        dout = torch.autograd.Variable(
            tensor([[[[0.3, -0.1], [0.03, 0.1]]]], dtype=dtype, device=device))

        result_torch.backward(dout)
        assert conv.is_manual[0] == 1

        result = result_torch.cpu().detach().numpy()
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.cpu().numpy(), cache)

        print("expected dx: ", expected_dx)
        print("expected dw: ", expected_dw)
        print("expected db: ", expected_db)

        # Are the gradients correct?

        self.logger.debug("\nexpected dx: " + str(expected_dx))
        self.logger.debug("\ncomputed dx: " + str(x_torch.grad.cpu().numpy()))

        np.testing.assert_array_almost_equal(x_torch.grad.cpu().detach().numpy(),
                                             expected_dx)

        self.logger.debug("\nexpected dw: " + str(expected_dw))
        self.logger.debug("\ncomputed dw: " + str(y_torch.grad.cpu().numpy()))

        np.testing.assert_array_almost_equal(y_torch.grad.cpu().detach().numpy(),
                                             expected_dw)

        np.testing.assert_array_almost_equal(b_torch.grad.cpu().detach().numpy(),
                                             expected_db)

    def test_FunctionBackwardNoCompressionConv2dfft2Images(self):
        x = np.array([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 1.0],
                        [1.0, -1.0, -2.0]]],
                      [[[2.0, 1.0, 3.0],
                        [4.0, 2.0, 1.0],
                        [-1.0, -2.0, -3.0]]]
                      ])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]]]])
        b = np.array([1.0])
        dtype = torch.float
        x_torch = tensor(x, requires_grad=True, dtype=dtype)
        print("size of x: ", x_torch.size())
        y_torch = tensor(y, requires_grad=True, dtype=dtype)
        b_torch = tensor(b, requires_grad=True, dtype=dtype)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        conv = Conv2dfft(weight_value=y_torch, bias_value=b_torch,
                         args=Arguments())
        result_torch = conv.forward(input=x_torch)
        dout = torch.autograd.Variable(tensor(
            [[[[0.1, -0.2], [0.01, 0.2]]], [[[0.3, -0.1], [0.03, 0.1]]]],
            dtype=dtype))
        result_torch.backward(dout)
        # ctx = MockContext()
        # Conv2dfftFunction.forward(ctx, input=x_torch, filter=y_torch, bias=None)

        result = result_torch.detach().numpy()
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.numpy(), cache)

        print("expected dx: ", expected_dx)
        print("expected dw: ", expected_dw)
        print("expected db: ", expected_db)

        # ctx.needs_input_grad = [True, True, True]
        # Conv2dfftFunction.backward(ctx, dout)
        assert conv.is_manual[0] == 1

        # Are the gradients correct?

        self.logger.debug("expected dx: " + str(expected_dx))
        self.logger.debug("computed dx: " + str(x_torch.grad))

        np.testing.assert_array_almost_equal(x_torch.grad.detach().numpy(),
                                             expected_dx)

        self.logger.debug("expected dw: " + str(expected_dw))
        self.logger.debug("computed dw: " + str(y_torch.grad))

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
                    second=item_w, delta=delta,
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
        # expected_result = np.array([[[[11.2787, 14.2694, 12.6907],
        #                               [14.0552, 15.6585, 12.3298],
        #                               [12.0275, 11.8809, 7.7573]]]])
        expected_result = np.array([[[[12.2992, 13.6678, 10.92],
                                      [15.9293, 16.679, 11.7282],
                                      [13.3441, 13.755, 8.7778]]]])
        conv = Conv2dfftFunction()

        x_torch = torch.tensor(data=x, requires_grad=True)
        y_torch = torch.tensor(data=y, requires_grad=True)
        b_torch = torch.tensor(data=b, requires_grad=True)

        out_size = 3

        result_torch = conv.forward(ctx=None, input=x_torch, filter=y_torch,
                                    bias=b_torch, out_size=out_size,
                                    args=Arguments())
        result = result_torch.detach().numpy()
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result, decimal=4,
            err_msg="Manual: Expected x is different from computed y.")

        # Prevent any interference/overlap of variables in manual and auto
        # differentiation.
        x_torch_auto = torch.tensor(data=x, requires_grad=True)
        y_torch_auto = torch.tensor(data=y, requires_grad=True)
        b_torch_auto = torch.tensor(data=b, requires_grad=True)

        convAuto = Conv2dfftAutograd(weight_value=y_torch_auto,
                                     bias=b_torch_auto,
                                     out_size=out_size)
        resultAuto = convAuto.forward(input=x_torch_auto)
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=resultAuto.cpu().detach().numpy(),
            decimal=4,
            err_msg="Auto: Expected x is different from computed y.")

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

        # approximate_expected_dx = np.array(
        #     [[[[0.0004, 0.1056, 0.1608, 0.1246, 0.0241],
        #        [0.0604, 0.1825, 0.1858, 0.0676, -0.0829],
        #        [0.1250, 0.1951, 0.1164, -0.0518, -0.1829],
        #        [0.1456, 0.1338, 0.0051, -0.1437, -0.2005],
        #        [0.1066, 0.0448, -0.0645, -0.1389, -0.1225]]]])

        approximate_expected_dx = np.array([[[
            [-0.0148, 0.0503, 0.1306, 0.1655, 0.1288],
            [0.1054, 0.1526, 0.1158, 0.0227, -0.0567],
            [0.1963, 0.2130, 0.0595, -0.1488, -0.2549],
            [0.1895, 0.1861, 0.0040, -0.2197, -0.3165],
            [0.0901, 0.0920, -0.0089, -0.1367, -0.1952]]]])

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
        # approximate_expected_dw = np.array([[[[1.1816, 1.8317],
        #                                       [1.5589, 1.4568]]]])
        approximate_expected_dw = np.array([[[[1.2042, 2.0410],
                                              [1.6021, 1.6371]]]])

        np.testing.assert_array_almost_equal(
            x=approximate_expected_dw, y=y_torch.grad, decimal=4,
            err_msg="Expected x is different from computed y.")

        self._check_delta2D(actual_result=y_torch.grad,
                            accurate_expected_result=expected_dw, delta=4.0)

        np.testing.assert_array_almost_equal(b_torch.grad,
                                             expected_db)

    def test_AutogradForwardNoCompression(self):
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        conv = Conv2dfftAutograd(weight_value=y, bias=b)
        result = conv.forward(input=x)
        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result.detach().numpy(),
            err_msg="The expected array x and computed y are not almost equal.")

    def test_AutogradForwardWithCompression(self):
        # A single input 2D map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        conv = Conv2dfftAutograd(weight_value=y, bias=b,
                                 args=Arguments(index_back=1,
                                                next_power2=False,
                                                preserve_energy=100))
        result = conv.forward(input=x)
        # expect = np.array([[[[21.5, 22.0], [17.5, 13.]]]])
        expect = np.array([[[[21.75, 21.75], [18.75, 13.75]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=result.detach().numpy(),
            err_msg="The expected array x and computed y are not almost equal")

    def test_FunctionForwardBackwardRandom(self):
        num_channels = 3
        num_data_points = 11
        input_H = 28
        input_W = 28
        filter_H = 5
        filter_W = 5
        num_filters = 5

        print("\nstart test:")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("cuda is available")
        else:
            device = torch.device("cpu")
            print("cuda is not available")

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
        x_torch = tensor(x, requires_grad=True, dtype=dtype, device=device)
        y_torch = tensor(y, requires_grad=True, dtype=dtype, device=device)
        b_torch = tensor(b, requires_grad=True, dtype=dtype, device=device)
        conv = Conv2dfftFunction()
        preserve_energy = 100.0
        is_manual = tensor([0])
        ctx = MockContext()
        result_torch = conv.forward(ctx=ctx, input=x_torch, filter=y_torch,
                                    bias=b_torch, is_manual=is_manual,
                                    args=Arguments(
                                        preserve_energy=preserve_energy))
        # dout = tensor(result/100.0, dtype=dtype)
        dout = torch.randn(result_torch.shape, dtype=dtype, device=device)

        ctx.needs_input_grad = [True, True, True]
        dx, dw, db, _, _, _, _, _, _ = conv.backward(ctx, dout)
        assert is_manual[0] == 1

        result = result_torch.cpu().detach().numpy()
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_array_almost_equal(
            x=np.array(expected_result), y=result, decimal=4,
            err_msg="The expected result x differs from the computed result y.")

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x, y, b, conv_param)

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout.cpu().detach().numpy(), cache)

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
            x=expected_dw, y=dw.cpu().detach().numpy(), decimal=4,
            err_msg="Expected x is different from computed y.")

        np.testing.assert_array_almost_equal(
            x=expected_dx, y=dx.cpu().detach().numpy(), decimal=5,
            err_msg="Expected x is different from computed y.")

        np.testing.assert_array_almost_equal(
            x=expected_db, y=db.cpu().detach().numpy(), decimal=4,
            err_msg="Expected x is different from computed y.")

    def test_ForwardNoCompressionForConv2dfftPreserveEenrgy(self):
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]])
        b = tensor([0.0])
        convManual = Conv2dfft(weight_value=y, bias=b,
                               args=Arguments(index_back=0,
                                              next_power2=True,
                                              preserve_energy=80))
        result = convManual.forward(input=x)
        print("result of manual convolution: ", result)
        expect_full_correct = np.array([[[[22.0, 22.0], [18., 14.]]]])
        print("expect_full_correct: ", expect_full_correct)

        # expect_approximate = np.array([[[[20.75, 22.25], [18.25, 12.75]]]])
        expect_approximate = np.array([[[[21.7500, 21.7500],
                                         [18.7500, 13.7500]]]])
        np.testing.assert_array_almost_equal(
            x=expect_approximate, y=result.detach().numpy(),
            err_msg="The expected array x and computed y are not almost equal.")

    def test_rfft_symmetry(self):
        x = tensor([[1.0, 2.0, 3.0],
                    [3.0, 4.0, 1.0],
                    [1.0, 2.0, 1.0]])
        xfft = torch.rfft(x, signal_ndim=2, onesided=False)
        print("xfft: ", xfft)
        self.check_DC_component(x, xfft)

    def check_DC_component(self, x, xfft):
        sum_x = x.sum()
        if xfft[0][0][0] != sum_x:
            raise Exception(
                "DC component is not at the position of (0,0) in the xfft!")
        if xfft[0][0][1] != 0.0:
            raise Exception(
                "The DC compoenent should be a real value without any imaginary part.")

    def test_rfft_DC_component_even(self):
        x = tensor([[1.0, 2.0, 3.0, 5.0],
                    [3.0, 4.0, 1.0, -1.0],
                    [1.0, 2.0, 1.0, 1.0],
                    [5.0, 3.0, 0.0, -1.0]])
        H, W = x.size()
        assert H == W
        xfft = torch.rfft(x, signal_ndim=2, onesided=False)
        print("xfft: ", xfft)
        Hfft, Wfft, ComplexDim = xfft.size()
        assert ComplexDim == 2
        assert Hfft == Wfft
        assert H == Hfft

        # check that the (0,0) coordinate of the xfft is the DC component
        # https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
        # F(0,0) represents the DC-component of the image which corresponds to
        # the average brightness
        self.check_DC_component(x, xfft)
        # F(N-1,N-1) represents the highest frequency

    def test_symmetries_even(self):
        x = tensor([[1.0, 2.0, 3.0, 5.0, -1.0, 2.0],
                    [3.0, 4.0, 1.0, -1.0, 3.0, 5.0],
                    [1.0, 2.0, 1.0, 1.0, 0.0, -2.0],
                    [5.0, 3.0, 0.0, -1.0, -1.0, 0.0],
                    [3.0, 0.0, 1.0, -1.0, 0.0, -2.0],
                    [-1.0, -2.0, 1.0, 1.0, 3.0, 1.0]])
        xfft = torch.rfft(x, signal_ndim=2, onesided=False)
        print("xfft: ", xfft)
        print("expected xfft: ", tensor(
            [[[40.0000, 0.0000],
              [9.0000, -6.9282],
              [4.0000, -1.7321],
              [6.0000, 0.0000],
              [4.0000, 1.7321],
              [9.0000, 6.9282]],

             [[13.0000, -12.1244],
              [-14.0000, -0.0000],
              [-8.0000, 10.3923],
              [-11.0000, 8.6603],
              [-2.0000, -3.4641],
              [-8.0000, -13.8564]],

             [[7.0000, -8.6603],
              [3.0000, -19.0526],
              [7.0000, 5.1962],
              [-9.0000, -1.7321],
              [7.0000, -1.7321],
              [3.0000, -5.1962]],

             [[-8.0000, 0.0000],
              [-11.0000, -13.8564],
              [10.0000, 1.7321],
              [-2.0000, 0.0000],
              [10.0000, -1.7321],
              [-11.0000, 13.8564]],

             [[7.0000, 8.6603],
              [3.0000, 5.1962],
              [7.0000, 1.7321],
              [-9.0000, 1.7321],
              [7.0000, -5.1962],
              [3.0000, 19.0526]],

             [[13.0000, 12.1244],
              [-8.0000, 13.8564],
              [-2.0000, 3.4641],
              [-11.0000, -8.6603],
              [-8.0000, -10.3923],
              [-14.0000, 0.0000]]]))
        print("xfft real part: ", xfft[:, :, 0])
        print("xfft imaginary part: ", xfft[:, :, 1])
        print("spectrum: ", get_spectrum(xfft))
        self.check_DC_component(x, xfft)
        H, W, _ = xfft.size()

        # Check the symmetries.
        KeepDim = W // 2
        middle_col = xfft[:, KeepDim]

        catted = torch.cat((xfft[:, KeepDim, 0], xfft[:, -KeepDim, 0]), dim=0)
        expect_middle_col_real = torch.mean(catted, dim=0)

    def test_symmetries_odd_even(self):
        x = tensor([[[[1.0, 2.0, 3.0, 5.0, -1.0],
                      [3.0, 4.0, 1.0, -1.0, 3.0],
                      [1.0, 2.0, 1.0, 1.0, 0.0],
                      [5.0, 3.0, 0.0, -1.0, -1.0],
                      [3.0, 0.0, 1.0, -1.0, 0.0]]]])
        N = 5
        M = 2
        KeepDim = M // 2
        xfft = torch.rfft(x, signal_ndim=2, onesided=False)
        print("xfft: ", xfft)
        print("xfft real part: ", xfft[..., 0])
        print("xfft imaginary part: ", xfft[..., 1])
        print("xfft spectrum: ", get_spectrum(xfft))

        xfft_one = torch.rfft(x, signal_ndim=2, onesided=True)
        print("xfft_one: ", xfft_one)
        print("xfft_one real part: ", xfft_one[..., 0])
        print("xfft_one imaginary part: ", xfft_one[..., 1])
        print("xfft_one spectrum: ", get_spectrum(xfft_one))

        print("xfft spectrum: ", get_spectrum(xfft))
        print("xfft_one spectrum: ", get_spectrum(xfft_one))

    def test_FunctionForwardCompressionConvFFTPreserveEnergyCifar10LeNet1stLayer(
            self):
        x = cifar10_image
        print("shape of the input image: ", x.size())
        y = cifar10_lenet_filter
        print("shape of the filter: ", y.size())
        b = torch.tensor([0.0])
        # get the expected results from numpy correlate

        # print("expected_result_numpy: ", expected_result_numpy)

        preserved_energies = [100., 99., 98.5, 98., 97., 96., 95., 94., 93.,
                              92., 91., 90., 89., 87., 85., 80., 70., 60., 50.,
                              40., 10., 5., 1.]
        # preserved_energies = [1.0]
        # indexes_back = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        expected_result_tensor = F.conv2d(input=x, weight=y, bias=b)

        for preserve_energy in preserved_energies:
            conv = Conv2dfft(weight_value=y,
                             bias_value=b,
                             args=Arguments(
                                 preserve_energy=preserve_energy,
                                 index_back=0,
                                 is_debug=True,
                                 next_power2=True,
                                 compress_type=CompressType.STANDARD))
            result = conv.forward(input=x)
            # print("actual result: ", result)

            result = result.float()
            abs_error = torch.sum(
                torch.abs(result - expected_result_tensor)).item()
            expected_total = torch.sum(torch.abs(expected_result_tensor))
            relative_error = abs_error / expected_total * 100.0
            # relative_error = torch.mean(torch.abs(result) / torch.abs(expected_result_tensor) * 100)
            print(f"absolute divergence for preserved energy,{preserve_energy}"
                  f",absolute error,{abs_error},"
                  f"relative error (%),{relative_error}")

    def test_FunctionForwardCompressionConvFFTIndexBackCifar10LeNet1stLayer(
            self):
        start = time.time()
        x = cifar10_image
        print("shape of the input image: ", x.size())
        y = cifar10_lenet_filter
        print("shape of the filter: ", y.size())
        b = torch.tensor([0.0])
        # get the expected results from numpy correlate

        expected_result_tensor = F.conv2d(input=x, weight=y, bias=b)
        N, C, H, W = x.size()
        K, C, HH, WW = y.size()
        out_size = H - HH + 1
        fft_size = H + out_size - 1
        half_fft_size = fft_size // 2 + 1
        fft_numel = half_fft_size * fft_size * C

        # for index_back in range(1, fft_numel, 10):
        for index_back in range(1, 2):
            print("index back: ", index_back)
            conv = Conv2dfft(weight_value=y,
                             bias_value=b,
                             args=Arguments(
                                 index_back=index_back,
                                 preserve_energy=100,
                                 is_debug=True,
                                 next_power2=False,
                                 compress_type=CompressType.STANDARD))
            result = conv.forward(input=x)
            # print("actual result: ", result)

            result = result.float()
            abs_error = torch.sum(
                torch.abs(result - expected_result_tensor)).item()
            print("abs error: ", abs_error)
            expected_total = torch.sum(
                torch.abs(expected_result_tensor) + torch.abs(result))
            relative_error = 100.0 * abs_error / expected_total
            print("relative error: ", relative_error)
            # relative_error = torch.mean(torch.abs(result) / torch.abs(expected_result_tensor) * 100)
            print(f"absolute divergence for index back,{index_back},"
                  f"absolute error,{abs_error},"
                  f"relative error (%),{relative_error}")
        print("elapsed: ", time.time() - start)

    def test_profile_forward_pass(self):
        """
        100 reps wtih/wihtout n parts of input xfft:

        without:
        shape of the filter:  torch.Size([1, 3, 5, 5])
        pytorch Conv2d forward (sec):  0.017931461334228516
        Conv2dfft forward (sec):  0.37201499938964844
        pytorch Conv2d backward (sec):  0.018948078155517578
        Conv2dfft backward (sec):  0.28224706649780273

        with (is it correct)?
        shape of the filter:  torch.Size([1, 3, 5, 5])
        pytorch Conv2d forward (sec):  0.010969877243041992
        Conv2dfft forward (sec):  0.293215274810791
        pytorch Conv2d backward (sec):  0.022943496704101562
        Conv2dfft backward (sec):  0.2832372188568115

        :return:
        """
        x = cifar10_image
        x.requires_grad_()
        print("shape of the input image: ", x.size())
        y = cifar10_lenet_filter
        y.requires_grad_()
        print("shape of the filter: ", y.size())
        # get the expected results from numpy correlate
        # print("expected_result_numpy: ", expected_result_numpy)
        # preserved_energies = [1.0]
        # indexes_back = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        x = x.to(device)
        y = y.to(device)

        dtype = torch.float
        x = x.to(dtype)
        y = y.to(dtype)

        repeat = 1
        convTorch = torch.nn.Conv2d(in_channels=y.shape[1],
                                    out_channels=y.shape[0],
                                    kernel_size=(y.shape[2], y.shape[3]))
        convTorch.to(device)
        start = time.time()
        for _ in range(repeat):
            expected_result_tensor = convTorch(input=x)
        print("pytorch Conv2d forward (sec): ", time.time() - start)

        preserve_energy = 99.0
        conv = Conv2dfft(in_channels=y.shape[1],
                         out_channels=y.shape[0],
                         kernel_size=(y.shape[2], y.shape[3]),
                         bias=False,
                         args=Arguments(
                             preserve_energy=preserve_energy,
                             index_back=0,
                             is_debug=False,
                             next_power2=True,
                             compress_type=CompressType.STANDARD))
        conv.to(device)
        start = time.time()
        for _ in range(repeat):
            result = conv.forward(input=x)
        print("Conv2dfft forward (sec): ", time.time() - start)

        dout = torch.randn(result.shape[0], result.shape[1], result.shape[2],
                           result.shape[3], device=device, dtype=dtype)

        start = time.time()
        for _ in range(repeat):
            expected_result_tensor.backward(dout, retain_graph=True)
        print("pytorch Conv2d backward (sec): ", time.time() - start)

        start = time.time()
        for _ in range(repeat):
            result.backward(dout, retain_graph=True)
        print("Conv2dfft backward (sec): ", time.time() - start)
        assert conv.is_manual[0] == 1
        # print("actual result: ", result)

        device = torch.device("cpu")
        result = result.to(device)
        expected_result_tensor = expected_result_tensor.to(device)

        result = result.float()
        abs_error = torch.sum(
            torch.abs(result - expected_result_tensor)).item()
        expected_total = torch.sum(
            torch.abs(expected_result_tensor) + torch.abs(result))
        relative_error = 100.0 * abs_error / expected_total
        # relative_error = torch.mean(torch.abs(result) / torch.abs(expected_result_tensor) * 100)
        print(f"absolute divergence for preserved energy,{preserve_energy}"
              f",absolute error,{abs_error},"
              f"relative error (%),{relative_error}")

    def test_profile_forward_backward_pass_random(self):
        """
        100 reps wtih/wihtout n parts of input xfft:

        without:
        shape of the filter:  torch.Size([1, 3, 5, 5])
        pytorch Conv2d forward (sec):  0.017931461334228516
        Conv2dfft forward (sec):  0.37201499938964844
        pytorch Conv2d backward (sec):  0.018948078155517578
        Conv2dfft backward (sec):  0.28224706649780273

        with (is it correct)?
        shape of the filter:  torch.Size([1, 3, 5, 5])
        pytorch Conv2d forward (sec):  0.010969877243041992
        Conv2dfft forward (sec):  0.293215274810791
        pytorch Conv2d backward (sec):  0.022943496704101562
        Conv2dfft backward (sec):  0.2832372188568115

        :return:
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        x = torch.randn(64, 16, 8, 8, requires_grad=True, device=device)
        print("shape of the input image: ", x.size())
        y = torch.randn(32, 16, 3, 3, requires_grad=True, device=device)
        print("shape of the filter: ", y.size())

        # get the expected results from numpy correlate
        # print("expected_result_numpy: ", expected_result_numpy)
        # preserved_energies = [1.0]
        # indexes_back = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        repeat = 1
        convTorch = torch.nn.Conv2d(in_channels=y.shape[1],
                                    out_channels=y.shape[0],
                                    kernel_size=(y.shape[2], y.shape[3]),
                                    bias=False)
        convTorch.to(device)
        start = time.time()
        for _ in range(repeat):
            expected_result_tensor = convTorch(input=x)
        print("pytorch Conv2d forward (sec): ", time.time() - start)

        preserve_energy = 90.0
        convFFT = Conv2dfft(in_channels=y.shape[1],
                            out_channels=y.shape[0],
                            kernel_size=(y.shape[2], y.shape[3]),
                            bias=False,
                            args=Arguments(preserve_energy=preserve_energy,
                                           is_debug=False, next_power2=True))
        convFFT.to(device)
        ctx = MockContext()
        start = time.time()
        for _ in range(repeat):
            # result = convFFT.forward(input=x)
            result = Conv2dfftFunction.forward(ctx, x, y, None)
        print("Conv2dfft forward (sec): ", time.time() - start)

        dout = torch.randn(result.shape[0], result.shape[1], result.shape[2],
                           result.shape[3], device=device)

        ctx.needs_input_grad = [True, True, True]
        start = time.time()
        for _ in range(repeat):
            expected_result_tensor.backward(dout, retain_graph=True)
        print("pytorch Conv2d backward (sec): ", time.time() - start)

        start = time.time()
        for _ in range(repeat):
            Conv2dfftFunction.backward(ctx, dout)
            # result.backward(dout, retain_graph=True)
        print("Conv2dfft backward (sec): ", time.time() - start)

        # print("actual result: ", result)

        result = result.float()
        abs_error = torch.sum(
            torch.abs(result - expected_result_tensor)).item()
        expected_total = torch.sum(
            torch.abs(expected_result_tensor) + torch.abs(result))
        relative_error = 100.0 * abs_error / expected_total
        # relative_error = torch.mean(torch.abs(result) / torch.abs(expected_result_tensor) * 100)
        print(f"absolute divergence for preserved energy,{preserve_energy}"
              f",absolute error,{abs_error},"
              f"relative error (%),{relative_error}")

    def testConvStride(self):
        x = tensor(
            [[[
                [1.0, 2.0, 0.0, 4.0, 0.0, 5.0, 1.0],
                [2.0, 2.0, 3.0, 2.0, 0.0, 1.0, 0.0],
                [3.0, -1.0, 1.0, 0.0, 1.0, 2.0, 0.0],
                [4.0, -1.0, 2.0, 1.0, 6.0, 0.0, -1.0],
                [0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0],
                [0.0, 2.0, -1.0, 1.0, 1.0, 1.0, 0.0],
                [2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 8.0]
            ]]])
        y = tensor([[
            [[1.0, 2.0, 3.0],
             [-1.0, -1.0, 5.0],
             [1.0, -2.0, 4.0]]]])
        b = tensor([0.0])

        convStandard = torch.nn.functional.conv2d(input=x, weight=y, stride=2)
        print("convStandard: ", convStandard)

        conv = Conv2dfftFunction()
        convFFT = conv.forward(ctx=None, input=x, filter=y, bias=b, stride=2,
                               args=Arguments(stride_type=StrideType.STANDARD))
        print("convFFT: ", convFFT)

        convSpectral = Conv2dfftFunction()
        convFFTSpectral = convSpectral.forward(
            ctx=None, input=x, filter=y, bias=b, stride=2,
            args=Arguments(stride_type=StrideType.SPECTRAL))
        print("convFFTSpectral: ", convFFTSpectral)

        np.testing.assert_array_almost_equal(
            x=convStandard, y=convFFT, decimal=5,
            err_msg="The expected array x and computed y are not almost equal.")

    def testConvStrideForwardBackward(self):
        dtype = torch.float
        x = tensor(
            [[[
                [1.0, 2.0, 0.0, 4.0, 0.0, 5.0, 1.0],
                [2.0, 2.0, 3.0, 2.0, 0.0, 1.0, 0.0],
                [3.0, -1.0, 1.0, 0.0, 1.0, 2.0, 0.0],
                [4.0, -1.0, 2.0, 1.0, 6.0, 0.0, -1.0],
                [0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0],
                [0.0, 2.0, -1.0, 1.0, 1.0, 1.0, 0.0],
                [2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 8.0]
            ]]], requires_grad=True, dtype=dtype)
        x_expect = x.clone().detach().requires_grad_(True)
        y = tensor([[
            [[1.0, 2.0, 3.0],
             [-1.0, -1.0, 5.0],
             [1.0, -2.0, 4.0]]]], requires_grad=True, dtype=dtype)
        y_expect = y.clone().detach().requires_grad_(True)
        b = tensor([1.0], requires_grad=True, dtype=dtype)
        b_expect = b.clone().detach().requires_grad_(True)

        convStandard = torch.nn.functional.conv2d(
            input=x_expect, weight=y_expect, bias=b_expect, stride=2)
        print("convStandard: ", convStandard)

        is_manual = tensor([0])
        conv = Conv2dfft(weight_value=y, bias_value=b, stride=2,
                         is_manual=is_manual,
                         args=Arguments(stride_type=StrideType.STANDARD))

        convFFT = conv.forward(input=x)
        dout_np = np.array([[[[0.1, -0.2, 0.3],
                              [-0.1, 0.1, 0.2],
                              [-0.2, 1.1, -1.2]]]])
        dout = torch.autograd.Variable(tensor(dout_np, dtype=dtype))

        # get the expected result from the backward pass
        convStandard.backward(dout)
        convFFT.backward(dout.clone())
        print("is_manual: ", is_manual[0])
        assert is_manual[0] == 1
        assert conv.is_manual[0] == 1

        print("convFFT: ", convFFT)
        np.testing.assert_array_almost_equal(
            x=convStandard.cpu().detach().numpy(),
            y=convFFT.cpu().detach().numpy(), decimal=5,
            err_msg="The expected array x and computed y are not almost equal.")

        print("pytorch's grad x: ", x_expect.grad)
        print("pytorch's grad y: ", y_expect.grad)
        print("pytorch's grad bias: ", b_expect.grad)

        print("fft grad x: ", x.grad)
        print("fft grad y: ", conv.weight.grad)
        print("fft grad b: ", conv.bias.grad)

        # Are the gradients correct?
        np.testing.assert_array_almost_equal(
            x=x_expect.grad, y=x.grad, decimal=4,
            err_msg="Expected x is different from computed y.")

        self._check_delta2D(actual_result=x.grad,
                            accurate_expected_result=x_expect.grad,
                            delta=0.0001)

        print("Expected fully correct dw from y_expect.grad: ", y_expect.grad)
        print("actual result for dw from y.grad: ", y.grad)

        np.testing.assert_array_almost_equal(
            x=y_expect.grad, y=conv.weight.grad, decimal=4,
            err_msg="Expected x is different from computed y.")

        self._check_delta2D(actual_result=conv.weight.grad,
                            accurate_expected_result=y_expect.grad,
                            delta=0.0001)

        np.testing.assert_array_almost_equal(
            x=b_expect.grad, y=conv.bias.grad, decimal=4,
            err_msg="Expected x is different from computed y for bias gradient.")

    def test_conv2d_picker(self):
        in_planes = 3
        out_planes = 64
        stride = 1
        args = Arguments(conv_type=ConvType.FFT2D)
        from cnns.nnlib.pytorch_layers.conv_picker import Conv
        conv = Conv(kernel_sizes=[3], in_channels=in_planes,
                    out_channels=[out_planes], strides=[stride],
                    padding=[1], args=args, is_bias=False).get_conv()
        result = conv.forward(torch.randn(8, 3, 32, 32))
        result.backward(torch.ones_like(result))
        assert conv.is_manual[0] == 1


if __name__ == '__main__':
    unittest.main()
