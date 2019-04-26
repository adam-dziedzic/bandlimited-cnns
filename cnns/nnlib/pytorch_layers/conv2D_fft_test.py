import logging
import unittest
import time
import numpy as np
import torch
from torch import tensor
from cnns.nnlib.layers import conv_forward_naive, conv_backward_naive
from cnns.nnlib.pytorch_layers.conv2D_fft \
    import Conv2dfftAutograd, Conv2dfftFunction, Conv2dfft
from cnns.nnlib.pytorch_layers.pytorch_utils import MockContext
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from cnns.nnlib.pytorch_layers.pytorch_utils import get_numpy
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.pytorch_layers.test_data.cifar10_image import cifar10_image
from cnns.nnlib.pytorch_layers.test_data.cifar10_lenet_filter import \
    cifar10_lenet_filter
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import ConvExecType
from cnns.nnlib.utils.arguments import Arguments


class TestPyTorchConv2d(unittest.TestCase):

    def setUp(self):
        print("\n")
        log_file = "pytorch_conv2D_reuse_map_fft.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")
        seed = 31
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("cuda is available")
            torch.cuda.manual_seed_all(seed)
        else:
            self.device = torch.device("cpu")
            print("cuda is not available")
            torch.manual_seed(seed)
        self.dtype = torch.float
        self.ERR_MESSAGE_ALL_CLOSE = "The expected array desired and " \
                                     "computed actual are not almost equal."

    def testAutogradForwardNoCompression(self):

        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=self.device, dtype=self.dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]], device=self.device,
                   dtype=self.dtype)
        b = tensor([0.0], device=self.device, dtype=self.dtype)

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
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=self.device, dtype=self.dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]], device=self.device,
                   dtype=self.dtype)
        b = tensor([0.0], device=self.device, dtype=self.dtype)
        convManual = Conv2dfft(weight_value=y, bias_value=b)
        result = convManual.forward(input=x)
        print("result of manual convolution: ", result)
        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])

        np.testing.assert_allclose(
            desired=expect, actual=result.cpu().detach().numpy(),
            rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test_FunctionForwardNoCompression(self):
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=self.device, dtype=self.dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]], device=self.device,
                   dtype=self.dtype)
        b = tensor([0.0], device=self.device, dtype=self.dtype)
        conv = Conv2dfftFunction()

        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              args=Arguments())
        print("Result of conv function: ", result)

        expect = np.array([[[[22.0, 22.0], [18., 14.]]]])
        np.testing.assert_allclose(
            desired=expect, actual=get_numpy(result), rtol=1e-6,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test_2_channels_2_filters(self):
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]],
                     [[1., 1., 2.], [2., 3., 1.], [2., -1., 3.]]]],
                   device=self.device, dtype=self.dtype)
        print("shape of x: ", x.size())
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]], [[-1.0, 2.0], [3.0, -2.0]]],
                    [[[-1.0, 1.0], [2.0, 3.0]], [[-2.0, 1.0], [1.0, -3.0]]]],
                   device=self.device, dtype=self.dtype)
        b = tensor([0.0, 0.0], device=self.device, dtype=self.dtype)
        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              args=Arguments())
        expect = np.array([[[[23.0, 32.0], [30., 4.]], [[11.0, 12.0],
                                                        [13.0, -11.0]]]])
        np.testing.assert_allclose(
            desired=expect, actual=get_numpy(result), rtol=1e-6,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test_bias(self):
        # A single 2D input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=self.device, dtype=self.dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]],
                   device=self.device, dtype=self.dtype)
        b = tensor([-1.0],
                   device=self.device, dtype=self.dtype)
        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b)
        expect = np.array([[[[21.0, 21.0], [17., 13.]]]])
        np.testing.assert_allclose(
            desired=expect, actual=get_numpy(result), rtol=1e-6,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)

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
            ]]], dtype=self.dtype, device=self.device)
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
        ], dtype=self.dtype, device=self.device)
        b = tensor([1.0, 0.0], device=self.device, dtype=self.dtype)
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
            x=expect, y=get_numpy(result), decimal=5,
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
            ]]], device=self.device, dtype=self.dtype)
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
        ], device=self.device, dtype=self.dtype)
        b = tensor([1.0, 0.0], device=self.device, dtype=self.dtype)
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
            x=expect, y=get_numpy(result), decimal=5,
            err_msg="The expected array x and computed y are not almost equal")

    def test_FunctionForwardRandom(self):
        num_channels = 3
        num_data_points = 32
        input_H = 32
        input_W = 32
        filter_H = 3
        filter_W = 3
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
        expect, _ = conv_forward_naive(x, y, b, conv_param)
        self.logger.debug("expected result: " + str(expect))

        x_torch = tensor(x, device=self.device, dtype=self.dtype)
        y_torch = tensor(y, device=self.device, dtype=self.dtype)
        b_torch = tensor(b, device=self.device, dtype=self.dtype)

        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=x_torch, filter=y_torch,
                              bias=b_torch)
        self.logger.debug("obtained result: " + str(result))
        np.testing.assert_allclose(
            desired=expect, actual=get_numpy(result), rtol=1e-6,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test_FunctionForwardRandomWithPytorch(self):
        num_channels = 3
        num_data_points = 32
        input_H = 32
        input_W = 32
        filter_H = 3
        filter_W = 3
        num_filters = 64
        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, input_H,
                           input_W)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, filter_H, filter_W)
        # Bias: one for each filter
        # b = np.random.rand(num_filters)
        b = np.zeros(num_filters)

        x_torch = tensor(x, device=self.device, dtype=self.dtype)
        y_torch = tensor(y, device=self.device, dtype=self.dtype)
        b_torch = tensor(b, device=self.device, dtype=self.dtype)

        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=x_torch, filter=y_torch,
                              bias=b_torch)
        # self.logger.debug("obtained result: " + str(result))

        expect = torch.nn.functional.conv2d(input=x_torch, weight=y_torch,
                                            bias=b_torch)
        # print("expect result from convStandard: ", expect)

        np.testing.assert_allclose(
            desired=get_numpy(expect), actual=get_numpy(result), rtol=1e-6,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test_FunctionForwardRandomWithPytorchWeights(self):
        num_channels = 3
        num_data_points = 32
        input_H = 32
        input_W = 32
        filter_H = 3
        filter_W = 3
        num_filters = 64

        # Input signal: 5 data points, 3 channels, 10 values.
        x = np.random.rand(num_data_points, num_channels, input_H,
                           input_W)
        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, filter_H, filter_W)
        # Bias: one for each filter
        # b = np.random.rand(num_filters)
        b = np.zeros(num_filters)

        x_torch = tensor(x, device=self.device, dtype=self.dtype)
        x_torch_clone = tensor(x, device=self.device, dtype=self.dtype)
        b_torch = tensor(b, device=self.device, dtype=self.dtype)

        convTorch = torch.nn.Conv2d(in_channels=y.shape[1],
                                    out_channels=y.shape[0],
                                    kernel_size=(y.shape[2], y.shape[3]),
                                    bias=False)

        # weight taken from torch's Conv2d
        weight = convTorch.weight.clone()
        weight = weight.requires_grad_(True)
        weight = weight.to(self.device)

        convTorch.to(self.device)
        expect = convTorch(input=x_torch_clone)

        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=x_torch, filter=weight, bias=None)

        np.testing.assert_allclose(
            desired=get_numpy(expect), actual=get_numpy(result), rtol=1e-4,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test_FunctionForwardRandomWithPytorchWeightsCifar10Image(self):
        num_channels = 3
        num_data_points = 32
        filter_H = 3
        filter_W = 3
        num_filters = 64

        x = cifar10_image
        # repeat the data point num_data_points times
        repetition = num_data_points
        while repetition > 1:
            x = torch.cat((x, x))
            repetition /= 2
        print("shape of the input batch: ", x.size())

        # Filters: 3 filters, 3 channels, 4 values.
        y = np.random.rand(num_filters, num_channels, filter_H, filter_W)
        # Bias: one for each filter
        # b = np.random.rand(num_filters)
        b = np.zeros(num_filters)

        x_torch = tensor(x, device=self.device, dtype=self.dtype)
        x_torch_clone = tensor(x, device=self.device, dtype=self.dtype)
        b_torch = tensor(b, device=self.device, dtype=self.dtype)

        convTorch = torch.nn.Conv2d(in_channels=y.shape[1],
                                    out_channels=y.shape[0],
                                    kernel_size=(y.shape[2], y.shape[3]),
                                    bias=False)

        # weight taken from torch's Conv2d
        weight = convTorch.weight.clone()
        weight = weight.requires_grad_(True)
        weight = weight.to(self.device)

        convTorch.to(self.device)
        expect = convTorch(input=x_torch_clone)

        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=x_torch, filter=weight, bias=None)

        np.testing.assert_allclose(
            desired=get_numpy(expect), actual=get_numpy(result), rtol=1e-5,
            err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test_FunctionBackwardNoCompressionWithBias(self):
        x = np.array([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 1.0],
                        [1.0, -1.0, -2.0]]]])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]]]])
        b = np.array([2.0])
        x_torch = tensor(x, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        y_torch = tensor(y, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        b_torch = tensor(b, requires_grad=True, dtype=self.dtype,
                         device=self.device)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        ctx = MockContext()
        args = Arguments()
        ctx.set_needs_input_grad(3)
        is_manual = tensor([0])
        result_torch = Conv2dfftFunction.forward(
            ctx, input=x_torch, filter=y_torch, bias=b_torch, args=args,
            is_manual=is_manual)
        result = get_numpy(result_torch)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        dout = tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=self.dtype,
                      device=self.device)

        dx, dw, db, _, _, _, _, _, _ = Conv2dfftFunction.backward(
            ctx, dout)

        dout_numpy = get_numpy(dout.clone())
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(dout_numpy, cache)

        self.logger.debug("\nexpected dx: " + str(expected_dx))
        self.logger.debug("\ncomputed dx: " + str(dx))
        assert is_manual[0] == 1

        # are the gradients correct
        np.testing.assert_array_almost_equal(get_numpy(dx),
                                             expected_dx)
        np.testing.assert_array_almost_equal(get_numpy(dw),
                                             expected_dw)
        np.testing.assert_array_almost_equal(get_numpy(db),
                                             expected_db)

    def test_FunctionBackwardNoCompressionNoBias(self):
        x = np.array([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 1.0],
                        [1.0, -1.0, -2.0]]]])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]]]])
        b = np.array([0.0])
        x_torch = tensor(x, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        y_torch = tensor(y, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        b_torch = tensor(b, requires_grad=True, dtype=self.dtype,
                         device=self.device)

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
        result = get_numpy(result_torch)
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        result_torch_2 = Conv2dfft(weight_value=y_torch, bias=b_torch).forward(
            input=x_torch)
        result2 = get_numpy(result_torch_2)
        print("actual result 2: ", result2)
        np.testing.assert_array_almost_equal(result2, np.array(expected_result))

        dout = torch.autograd.Variable(
            tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=self.dtype,
                   device=self.device))
        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(get_numpy(dout.clone()), cache)

        dx, dw, db, _, _, _, _, _, _ = Conv2dfftFunction.backward(ctx, dout)

        assert is_manual[0] == 1

        self.logger.debug("\nexpected dx: " + str(expected_dx))
        self.logger.debug("\ncomputed dx: " + str(dx))

        # are the gradients correct
        np.testing.assert_array_almost_equal(get_numpy(dx),
                                             expected_dx)
        np.testing.assert_array_almost_equal(get_numpy(dw),
                                             expected_dw)
        np.testing.assert_array_almost_equal(get_numpy(db),
                                             expected_db)

    def test_FunctionBackwardNoCompressionConv2dfft(self):
        x = np.array([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 1.0],
                        [1.0, -1.0, -2.0]]]])
        y = np.array([[[[2.0, 1.0], [-1.0, 2.0]]]])
        b = np.array([0.0])

        x_torch = tensor(x, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        y_torch = tensor(y, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        b_torch = tensor(b, requires_grad=True, dtype=self.dtype,
                         device=self.device)

        conv = Conv2dfft(weight_value=y_torch, bias_value=b_torch)
        # conv = Conv2dfft()

        result_torch = conv.forward(input=x_torch)
        dout = torch.autograd.Variable(
            tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=self.dtype,
                   device=self.device))
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

        x_torch = tensor(x, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        print("x size: ", x_torch.size())
        y_torch = tensor(y, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        print("y size: ", y_torch.size())
        b_torch = tensor(b, requires_grad=True, dtype=self.dtype,
                         device=self.device)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        conv = Conv2dfft(weight_value=y_torch, bias_value=b_torch)

        result_torch = conv.forward(input=x_torch)
        # dout = torch.autograd.Variable(
        #     tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=self.dtype))
        dout = tensor([[[[0.1, -0.2], [0.3, -0.1]]]], dtype=self.dtype,
                      device=self.device)
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
        x_torch = tensor(x, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        y_torch = tensor(y, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        b_torch = tensor(b, requires_grad=True, dtype=self.dtype,
                         device=self.device)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        conv = Conv2dfft(weight_value=y_torch, bias_value=b_torch)

        result_torch = conv.forward(input=x_torch)
        dout = torch.autograd.Variable(
            tensor([[[[0.3, -0.1], [0.03, 0.1]]]], dtype=self.dtype,
                   device=self.device))

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

        np.testing.assert_array_almost_equal(
            x_torch.grad.cpu().detach().numpy(),
            expected_dx)

        self.logger.debug("\nexpected dw: " + str(expected_dw))
        self.logger.debug("\ncomputed dw: " + str(y_torch.grad.cpu().numpy()))

        np.testing.assert_array_almost_equal(
            y_torch.grad.cpu().detach().numpy(),
            expected_dw)

        np.testing.assert_array_almost_equal(
            b_torch.grad.cpu().detach().numpy(),
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
        x_torch = tensor(x, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        print("size of x: ", x_torch.size())
        y_torch = tensor(y, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        b_torch = tensor(b, requires_grad=True, dtype=self.dtype,
                         device=self.device)

        conv_param = {'pad': 0, 'stride': 1}
        expected_result, cache = conv_forward_naive(x=x, w=y, b=b,
                                                    conv_param=conv_param)

        print("expected result: ", expected_result)

        conv = Conv2dfft(weight_value=y_torch, bias_value=b_torch,
                         args=Arguments())
        conv.to(self.device)

        result_torch = conv.forward(input=x_torch)
        dout = torch.autograd.Variable(tensor(
            [[[[0.1, -0.2], [0.01, 0.2]]], [[[0.3, -0.1], [0.03, 0.1]]]],
            dtype=self.dtype, device=self.device))
        result_torch.backward(dout)

        # ctx.needs_input_grad = [True, True, True]
        # Conv2dfftFunction.backward(ctx, dout)
        assert conv.is_manual[0] == 1
        # ctx = MockContext()
        # Conv2dfftFunction.forward(ctx, input=x_torch, filter=y_torch, bias=None)

        result = get_numpy(result_torch)
        print("actual result: ", result)
        np.testing.assert_array_almost_equal(result, np.array(expected_result))

        # get the expected result from the backward pass
        expected_dx, expected_dw, expected_db = \
            conv_backward_naive(get_numpy(dout), cache)

        print("expected dx: ", expected_dx)
        print("expected dw: ", expected_dw)
        print("expected db: ", expected_db)

        # Are the gradients correct?

        self.logger.debug("expected dx: " + str(expected_dx))
        self.logger.debug("computed dx: " + str(x_torch.grad))

        x_torch_grad = get_numpy(x_torch.grad)
        np.testing.assert_array_almost_equal(x_torch_grad, expected_dx)

        self.logger.debug("expected dw: " + str(expected_dw))
        self.logger.debug("computed dw: " + str(y_torch.grad))

        y_torch_grad = get_numpy(y_torch.grad)
        np.testing.assert_array_almost_equal(y_torch_grad, expected_dw)

        b_torch_grad = get_numpy(b_torch.grad)
        np.testing.assert_array_almost_equal(b_torch_grad, expected_db)

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

    def test_FunctionForwardBackwardRandom(self):
        num_channels = 3
        num_data_points = 11
        input_H = 28
        input_W = 28
        filter_H = 5
        filter_W = 5
        num_filters = 5

        print("\nstart test:")

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
        x_torch = tensor(x, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        y_torch = tensor(y, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        b_torch = tensor(b, requires_grad=True, dtype=self.dtype,
                         device=self.device)
        conv = Conv2dfftFunction()
        preserve_energy = 100.0
        is_manual = tensor([0])
        ctx = MockContext()
        result_torch = conv.forward(ctx=ctx, input=x_torch, filter=y_torch,
                                    bias=b_torch, is_manual=is_manual,
                                    args=Arguments(
                                        preserved_energy=preserve_energy))
        # dout = tensor(result/100.0, dtype=self.dtype)
        dout = torch.randn(result_torch.shape, device=self.device)

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

    def test_FunctionForwardBackwardRandomCompareToPytorch(self):
        print("\nstart test:")

        C = 3  # nr of channels
        F = 64  # nr or filters
        K = 3  # kernel size
        N = 32  # nr of input maps
        H = 32  # height
        W = 32  # width

        input_fft = torch.randn(N, C, H, W, dtype=self.dtype,
                                device=self.device,
                                requires_grad=True)
        input_torch = input_fft.clone().requires_grad_()

        conv_torch = torch.nn.Conv2d(
            in_channels=C, out_channels=F, kernel_size=K)
        conv_torch.to(self.device)

        is_manual = tensor([0])
        conv_fft = Conv2dfft(weight_value=conv_torch.weight.clone(),
                             bias_value=conv_torch.bias.clone(),
                             is_manual=is_manual)

        result_torch = conv_torch.forward(input=input_torch)
        result_fft = conv_fft.forward(input=input_fft)

        dout = torch.randn(result_fft.shape, dtype=self.dtype,
                           device=self.device)

        result_fft.backward(dout)
        result_torch.backward(dout.clone())
        assert is_manual[0] == 1

        dx_expect = input_torch.grad.cpu().detach().numpy()
        dy_expect = conv_torch.weight.grad.cpu().numpy()
        db_expect = conv_torch.bias.grad.cpu().numpy()

        dx = get_numpy(input_fft.grad)
        dy = get_numpy(conv_fft.weight.grad)
        db = get_numpy(conv_fft.bias.grad)

        result_torch = result_fft.cpu().detach().numpy()
        result_fft = result_torch.cpu().detach().numpy()

        np.testing.assert_allclose(
            desired=result_torch, actual=result_fft,
            rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

        # are the gradients correct
        np.testing.assert_allclose(
            desired=dx_expect, actual=dx,
            rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

        np.testing.assert_allclose(
            desired=dy_expect, actual=dy,
            rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

        np.testing.assert_allclose(
            desired=db_expect, actual=db,
            rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def test_ForwardCompressionForConv2dfftPreserveEenrgy(self):
        # Don't use next power of 2.
        # A single input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=self.device, dtype=self.dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]], device=self.device,
                   dtype=self.dtype)
        b = tensor([0.0], device=self.device, dtype=self.dtype)
        convManual = Conv2dfft(weight_value=y, bias=b,
                               args=Arguments(next_power2=True,
                                              preserved_energy=80))
        result = convManual.forward(input=x)
        print("result of manual convolution: ", result)
        expect_full_correct = np.array([[[[22.0, 22.0], [18., 14.]]]])
        print("expect_full_correct: ", expect_full_correct)

        # expect_approximate = np.array([[[[20.75, 22.25], [18.25, 12.75]]]])
        expect_approximate = np.array([[[[21.7500, 21.7500],
                                         [18.7500, 13.7500]]]])

        np.testing.assert_allclose(
            desired=expect_approximate, actual=result.cpu().detach().numpy(),
            rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

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

        """
        seed = 31
        torch.cuda.manual_seed_all(seed)
        x = cifar10_image
        x.requires_grad_()
        print("shape of the input image: ", x.size())
        y = cifar10_lenet_filter
        y.requires_grad_()
        print("shape of the filter: ", y.size())
        # get the expected results from numpy correlate
        # print("expected_result_numpy: ", expected_result_numpy)
        # preserved_energies = [1.0]
        # compress_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
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

        preserve_energy = 100.0
        compress_rate = 0.0

        args = Arguments(
            preserved_energy=preserve_energy,
            compress_rate=compress_rate,
            is_debug=False,
            next_power2=True,
            compress_type=CompressType.STANDARD)
        args.dtype = self.dtype

        conv = Conv2dfft(in_channels=y.shape[1],
                         out_channels=y.shape[0],
                         kernel_size=(y.shape[2], y.shape[3]),
                         bias=False,
                         args=args)
        conv.to(device)
        start = time.time()
        for _ in range(repeat):
            result = conv.forward(input=x)
        print("Conv2dfft forward (sec): ", time.time() - start)

        dout = torch.randn(result.shape[0], result.shape[1], result.shape[2],
                           result.shape[3], device=self.device,
                           dtype=self.dtype)

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
        100 reps wtith/without n parts of input xfft:

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
            print("CUDA device is available")
        else:
            device = torch.device('cpu')
            print("CUDA device is not available")
        x = torch.randn(32, 16, 8, 8, requires_grad=True, device=self.device)
        print("shape of the input image: ", x.size())
        y = torch.randn(64, 16, 3, 3, requires_grad=True, device=self.device)
        print("shape of the filter: ", y.size())

        # get the expected results from numpy correlate
        # print("expected_result_numpy: ", expected_result_numpy)
        # preserved_energies = [1.0]
        # compress_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256]

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

        preserve_energy = 100.0
        convFFT = Conv2dfft(in_channels=y.shape[1],
                            out_channels=y.shape[0],
                            kernel_size=(y.shape[2], y.shape[3]),
                            bias=False,
                            args=Arguments(preserved_energy=preserve_energy,
                                           is_debug=False, next_power2=True))
        convFFT.to(device)
        ctx = MockContext()

        # Zero the gradients before running the backward pass.
        convFFT.zero_grad()  # set gradients of all model params to zero

        start = time.time()
        for _ in range(repeat):
            # result = convFFT.forward(input=x)
            result = Conv2dfftFunction.forward(ctx, x, y, None)
        print("Conv2dfft forward (sec): ", time.time() - start)

        dout = torch.randn(result.shape[0], result.shape[1], result.shape[2],
                           result.shape[3], device=self.device)

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

    def test_correctness_forward_backward_pass_resnet18(self):
        """
        ResNet: size of the layers.

        input size:  torch.Size([32, 3, 32, 32])
        input size:  torch.Size([32, 64, 16, 16])
        input size:  torch.Size([32, 64, 16, 16])
        input size:  torch.Size([32, 64, 16, 16])
        input size:  torch.Size([32, 128, 8, 8])
        input size:  torch.Size([32, 128, 8, 8])
        input size:  torch.Size([32, 128, 8, 8])
        input size:  torch.Size([32, 256, 4, 4])
        input size:  torch.Size([32, 256, 4, 4])
        input size:  torch.Size([32, 512, 2, 2])
        input size:  torch.Size([32, 512, 2, 2])

        :return:
        """
        print("\n")
        # N, F, C, H, W, HH, WW
        N = 1  # number of data points in the batch is always the same
        HH = 3  # the size of the filter is always 3x3
        WW = 3
        # F, C, H, W
        # layers = [(64, 3, 32, 32),
        #           (64, 64, 16, 16),
        #           (128, 64, 16, 16),
        #           (128, 128, 8, 8),
        #           (256, 128, 8, 8),
        #           (256, 256, 4, 4),
        #           (512, 256, 4, 4),
        #           (512, 512, 2, 2)]

        # layers = [(512, 512, 2, 2)]
        # layers = [(64, 3, 32, 32)]
        layers = [(1, 1, 2, 2)]

        # N, F, C, H, W, HH, WW = 32, 64, 16, 8, 8, 3, 3
        # N, F, C, H, W, HH, WW = 32, 64, 3, 32, 32, 3, 3
        # N, F, C, H, W, HH, WW = 1, 1, 1, 8, 8, 3, 3
        # N, F, C, H, W, HH, WW = 1, 4, 1, 3, 3, 3, 3

        for layer in layers:
            F, C, H, W = layer

            num_data_points = N
            num_channels = C
            input_H = H
            input_W = W
            num_filters = F
            filter_H = HH
            filter_W = WW

            is_numpy_initialize = True

            if is_numpy_initialize:
                # Input signal: 5 data points, 3 channels, 10 values.
                x = np.random.rand(num_data_points, num_channels, input_H,
                                   input_W)
                # Filters: 3 filters, 3 channels, 4 values.
                y = np.random.rand(num_filters, num_channels, filter_H,
                                   filter_W)
                # Bias: one for each filter
                # b = np.random.rand(num_filters)
                b = np.zeros(num_filters)

                x = tensor(x, device=self.device, dtype=self.dtype,
                           requires_grad=True)
                x_clone = tensor(x, device=self.device, dtype=self.dtype,
                                 requires_grad=True)
                y = tensor(y, device=self.device, dtype=self.dtype)
                b = tensor(b, device=self.device, dtype=self.dtype)

            else:
                # Initialization in torch.
                x = torch.randn(N, C, H, W, requires_grad=True,
                                device=self.device,
                                dtype=self.dtype)
                x_clone = x.clone()
                print("shape of the input image: ", x.size())
                y = torch.randn(F, C, HH, WW, requires_grad=True,
                                device=self.device,
                                dtype=self.dtype)
                print("shape of the filter: ", y.size())

            # get the expected results from numpy correlate
            # print("expected_result_numpy: ", expected_result_numpy)
            # preserved_energies = [1.0]
            # compress_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256]

            repeat = 1
            convTorch = torch.nn.Conv2d(in_channels=y.shape[1],
                                        out_channels=y.shape[0],
                                        kernel_size=(y.shape[2], y.shape[3]),
                                        bias=False, padding=1)

            weight = convTorch.weight.clone()
            weight = weight.requires_grad_(True)

            convTorch.to(self.device)

            start = time.time()
            for _ in range(repeat):
                expected_result_tensor = convTorch(input=x_clone)
            print("pytorch Conv2d forward (sec): ", time.time() - start)

            preserve_energy = None
            convFFT = Conv2dfft(weight_value=weight, bias=False, padding=1,
                                args=Arguments(preserved_energy=preserve_energy,
                                               is_debug=False,
                                               next_power2=True))
            convFFT.to(self.device)
            ctx = MockContext()
            start = time.time()
            for _ in range(repeat):
                # result = convFFT.forward(input=x)
                result = Conv2dfftFunction.forward(ctx, x,
                                                   weight.to(self.device), None,
                                                   padding=1)
            print("Conv2dfft forward (sec): ", time.time() - start)

            dout = torch.randn(result.shape[0], result.shape[1],
                               result.shape[2],
                               result.shape[3], device=self.device,
                               dtype=self.dtype)

            ctx.needs_input_grad = [True, True, True]
            start = time.time()
            for _ in range(repeat):
                expected_result_tensor.backward(dout, retain_graph=True)
            print("pytorch Conv2d backward (sec): ", time.time() - start)

            start = time.time()
            for _ in range(repeat):
                dx, dw, db, _, _, _, _, _, _ = Conv2dfftFunction.backward(ctx,
                                                                          dout)
                # result.backward(dout, retain_graph=True)
            print("Conv2dfft backward (sec): ", time.time() - start)

            # print("actual result: ", result)

            abs_error = torch.sum(
                torch.abs(result - expected_result_tensor)).item()
            expected_total = torch.sum(
                torch.abs(expected_result_tensor) + torch.abs(result))
            relative_error = 100.0 * abs_error / expected_total
            # relative_error = torch.mean(torch.abs(result) / torch.abs(expected_result_tensor) * 100)
            print(f"absolute divergence for preserved energy,{preserve_energy}"
                  f",absolute error,{abs_error},"
                  f"relative error (%),{relative_error}")

            dx_expect = x_clone.grad
            dw_expect = convTorch.weight.grad

            torch.set_printoptions(threshold=5000, precision=6)

            print("expected dx: ", dx_expect)
            print("computed dx: ", dx)

            # print("expected dw: ", dw_expect)
            # print("computed dw: ", dw)

            torch.set_printoptions(threshold=1000)

            # move the tensors to numpy arrays on cpu
            dx_expect = get_numpy(dx_expect)
            dx = get_numpy(dx)

            dw_expect = get_numpy(dw_expect)
            dw = get_numpy(dw)

            expect = get_numpy(expected_result_tensor)
            result = get_numpy(result)

            try:
                np.testing.assert_allclose(
                    desired=expect, actual=result,
                    rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)
            except AssertionError as ex:
                print("Error for the forward result of convolution: ", ex)

            try:
                np.testing.assert_allclose(
                    desired=dx_expect, actual=dx,
                    rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)
            except AssertionError as ex:
                print("\nError for the gradients for the input: ", ex)

            try:
                np.testing.assert_allclose(
                    desired=dw_expect, actual=dw,
                    rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)
            except AssertionError as ex:
                print("\nError for the gradients for the weights: ", ex)

    def testConvStride(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
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
            ]]], device=self.device, dtype=self.dtype)
        y = tensor([[
            [[1.0, 2.0, 3.0],
             [-1.0, -1.0, 5.0],
             [1.0, -2.0, 4.0]]]], device=self.device, dtype=self.dtype)
        b = tensor([0.0], device=self.device, dtype=self.dtype)

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

        np.testing.assert_allclose(
            desired=convStandard.cpu().detach().numpy(),
            actual=convFFT.cpu().detach().numpy(),
            rtol=1e-6, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

    def testConvStrideForwardBackward(self):
        x = tensor(
            [[[
                [1.0, 2.0, 0.0, 4.0, 0.0, 5.0, 1.0],
                [2.0, 2.0, 3.0, 2.0, 0.0, 1.0, 0.0],
                [3.0, -1.0, 1.0, 0.0, 1.0, 2.0, 0.0],
                [4.0, -1.0, 2.0, 1.0, 6.0, 0.0, -1.0],
                [0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0],
                [0.0, 2.0, -1.0, 1.0, 1.0, 1.0, 0.0],
                [2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 8.0]
            ]]], requires_grad=True, dtype=self.dtype)
        x_expect = x.clone()
        y = tensor([[
            [[1.0, 2.0, 3.0],
             [-1.0, -1.0, 5.0],
             [1.0, -2.0, 4.0]]]], requires_grad=True, dtype=self.dtype)
        y_expect = y.clone()
        b = tensor([1.0], requires_grad=True, dtype=self.dtype)
        b_expect = b.clone()

        x = x.to(self.device)
        x_expect = x_expect.to(self.device)
        y = y.to(self.device)
        y_expect = y_expect.to(self.device)
        b = b.to(self.device)
        b_expect = b_expect.to(self.device)

        conv_torch = torch.nn.functional.conv2d(
            input=x_expect, weight=y_expect, bias=b_expect, stride=2)
        conv_torch = conv_torch.to(self.device)
        # print("convStandard: ", conv_torch)

        is_manual = tensor([0])
        conv = Conv2dfft(weight_value=y, bias_value=b, stride=2,
                         is_manual=is_manual,
                         args=Arguments(stride_type=StrideType.STANDARD))
        conv = conv.to(self.device)

        convFFT = conv.forward(input=x)

        dout_torch = tensor([[[[0.1, -0.2, 0.3],
                               [-0.1, 0.1, 0.2],
                               [-0.2, 1.1, -1.2]]]], dtype=self.dtype,
                            device=self.device)
        dout_fft = dout_torch.clone()

        dout_torch = dout_torch.to(self.device)
        dout_fft = dout_fft.to(self.device)

        # get the expected result from the backward pass
        conv_torch.backward(dout_torch)
        convFFT.backward(dout_fft)

        print("is_manual: ", is_manual[0])
        assert is_manual[0] == 1
        assert conv.is_manual[0] == 1

        print("convFFT: ", convFFT)
        np.testing.assert_array_almost_equal(
            x=conv_torch.cpu().detach().numpy(),
            y=convFFT.cpu().detach().numpy(), decimal=5,
            err_msg="The expected array x and computed y are not almost equal.")

        x_expect = x_expect.cpu()
        y_expect = y_expect.cpu()
        b_expect = b_expect.cpu()

        x = x.cpu()
        conv.to(torch.device("cpu"))

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
        args.dtype = self.dtype
        from cnns.nnlib.pytorch_layers.conv_picker import Conv
        conv = Conv(kernel_sizes=[3], in_channels=in_planes,
                    out_channels=[out_planes], strides=[stride],
                    padding=[1], args=args, is_bias=False).get_conv()
        conv.to(self.device)
        result = conv.forward(
            torch.randn(8, 3, 32, 32, device=self.device, dtype=self.dtype))
        result.backward(torch.ones_like(result, device=self.device,
                                        dtype=self.dtype))
        assert conv.is_manual[0] == 1

    def test_correctness_forward_backward_pass_for_all_conv_modes(self):
        print("\n")

        N, F, C, H, W, HH, WW = 32, 64, 16, 8, 8, 3, 3
        # N, F, C, H, W, HH, WW = 1, 1, 1, 8, 8, 3, 3
        # N, F, C, H, W, HH, WW = 1, 4, 1, 3, 3, 3, 3

        num_data_points = N
        num_channels = C
        input_H = H
        input_W = W
        num_filters = F
        filter_H = HH
        filter_W = WW

        conv_exec_types = [ConvExecType.CUDA,
                           ConvExecType.CUDA_SHARED_LOG,
                           ConvExecType.CUDA_DEEP,
                           ConvExecType.BATCH,
                           ConvExecType.SERIAL]

        for conv_exec_type in conv_exec_types:

            print("conv exec type: ", conv_exec_type.name)

            is_numpy_initialize = True
            padding = 1

            if is_numpy_initialize:
                np.random.seed(31)
                # Input signal: 5 data points, 3 channels, 10 values.
                x = np.random.rand(num_data_points, num_channels, input_H,
                                   input_W)
                # Filters: 3 filters, 3 channels, 4 values.
                y = np.random.rand(num_filters, num_channels, filter_H,
                                   filter_W)
                # Bias: one for each filter
                # b = np.random.rand(num_filters)
                b = np.zeros(num_filters)

                x = tensor(x, device=self.device, dtype=self.dtype,
                           requires_grad=True)
                x_clone = tensor(x, device=self.device, dtype=self.dtype,
                                 requires_grad=True)
                y = tensor(y, device=self.device, dtype=self.dtype)
                b = tensor(b, device=self.device, dtype=self.dtype)

            else:
                # Initialization in torch.
                x = torch.randn(N, C, H, W, requires_grad=True,
                                device=self.device,
                                dtype=self.dtype)
                x_clone = x.clone()
                print("shape of the input image: ", x.size())
                y = torch.randn(F, C, HH, WW, requires_grad=True,
                                device=self.device,
                                dtype=self.dtype)
                print("shape of the filter: ", y.size())

            # get the expected results from numpy correlate
            # print("expected_result_numpy: ", expected_result_numpy)
            # preserved_energies = [1.0]
            # compress_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256]

            repeat = 1
            convTorch = torch.nn.Conv2d(in_channels=y.shape[1],
                                        out_channels=y.shape[0],
                                        kernel_size=(y.shape[2], y.shape[3]),
                                        bias=False, padding=padding)

            weight = convTorch.weight.clone()
            weight = weight.requires_grad_(True)

            convTorch.to(self.device)

            start = time.time()
            for _ in range(repeat):
                expected_result_tensor = convTorch(input=x_clone)
            print("pytorch Conv2d forward (sec): ", time.time() - start)

            preserve_energy = 100.0
            convFFT = Conv2dfft(weight_value=weight, bias=False,
                                args=Arguments(preserved_energy=preserve_energy,
                                               is_debug=False, next_power2=True,
                                               conv_exec_type=conv_exec_type))
            convFFT.to(self.device)
            ctx = MockContext()
            start = time.time()
            for _ in range(repeat):
                # result = convFFT.forward(input=x)
                result = Conv2dfftFunction.forward(
                    ctx, input=x, filter=weight.to(self.device), bias=None,
                    padding=padding)
            print("Conv2dfft forward (sec): ", time.time() - start)

            dout = torch.randn(result.shape[0], result.shape[1],
                               result.shape[2],
                               result.shape[3], device=self.device,
                               dtype=self.dtype)

            ctx.needs_input_grad = [True, True, True]
            start = time.time()
            for _ in range(repeat):
                expected_result_tensor.backward(dout, retain_graph=True)
            print("pytorch Conv2d backward (sec): ", time.time() - start)

            start = time.time()
            for _ in range(repeat):
                dx, dw, db, _, _, _, _, _, _ = Conv2dfftFunction.backward(ctx,
                                                                          dout)
                # result.backward(dout, retain_graph=True)
            print("Conv2dfft backward (sec): ", time.time() - start)

            # print("actual result: ", result)

            abs_error = torch.sum(
                torch.abs(result - expected_result_tensor)).item()
            expected_total = torch.sum(
                torch.abs(expected_result_tensor) + torch.abs(result))
            relative_error = 100.0 * abs_error / expected_total
            # relative_error = torch.mean(torch.abs(result) / torch.abs(expected_result_tensor) * 100)
            print(f"absolute divergence for preserved energy,{preserve_energy}"
                  f",absolute error,{abs_error},"
                  f"relative error (%),{relative_error}")

            dx_expect = x_clone.grad
            dw_expect = convTorch.weight.grad

            # torch.set_printoptions(threshold=5000, precision=6)
            #
            # print("expected dx: ", dx_expect)
            # print("computed dx: ", dx)
            #
            # print("expected dw: ", dw_expect)
            # print("computed dw: ", dw)
            #
            # torch.set_printoptions(threshold=1000)

            # move the tensors to numpy arrays on cpu
            dx_expect = get_numpy(dx_expect)
            dx = get_numpy(dx)

            dw_expect = get_numpy(dw_expect)
            dw = get_numpy(dw)

            expect = get_numpy(expected_result_tensor)
            result = get_numpy(result)

            rtol = 1e-1

            np.testing.assert_allclose(
                desired=expect, actual=result,
                rtol=rtol, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

            np.testing.assert_allclose(
                desired=dx_expect, actual=dx,
                rtol=rtol, err_msg=self.ERR_MESSAGE_ALL_CLOSE)

            np.testing.assert_allclose(
                desired=dw_expect, actual=dw,
                rtol=rtol, err_msg=self.ERR_MESSAGE_ALL_CLOSE)


if __name__ == '__main__':
    unittest.main()
