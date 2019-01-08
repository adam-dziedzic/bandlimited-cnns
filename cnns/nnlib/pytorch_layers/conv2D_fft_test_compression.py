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


class TestPyTorchConv2dCompression(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv2D_reuse_map_fft.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("cuda is available")
        else:
            self.device = torch.device('cpu')
            print("no cuda device is available")
        self.dtype = torch.float
        self.ERR_MESSAGE_ALL_CLOSE = "The expected array desired and " \
                                     "computed actual are not almost equal."

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

    def test_FunctionForwardCompressionConvFFTPreserveEnergyCifar10LeNet1stLayer(
            self):
        print("\n")
        x = cifar10_image
        print("shape of the input image: ", x.size())
        y = cifar10_lenet_filter
        print("shape of the filter: ", y.size())
        b = torch.tensor([0.0])
        # get the expected results from numpy correlate

        # print("expected_result_numpy: ", expected_result_numpy)

        preserved_energies = [100., 99., 98.5, 98., 97., 96., 95., 94., 93.,
                              92., 91., 90., 89., 87., 85., 80., 70., 60.,
                              50.,
                              40., 10., 5., 1.]
        # preserved_energies = [1.0]
        # compress_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256]

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
            print(
                f"absolute divergence for preserved energy,{preserve_energy}"
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

        # for compress_rate in range(1, fft_numel, 10):
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

    def test_FunctionForwardCompression(self):
        # A single 2D input map.
        x = tensor([[[[1.0, 2.0, 3.0], [3.0, 4.0, 1.0], [1., 2., 1.]]]],
                   device=self.device, dtype=self.dtype)
        # A single filter.
        y = tensor([[[[1.0, 2.0], [3.0, 2.0]]]], device=self.device,
                   dtype=self.dtype)
        b = tensor([0.0], device=self.device, dtype=self.dtype)
        conv = Conv2dfftFunction()
        result = conv.forward(ctx=None, input=x, filter=y, bias=b,
                              args=Arguments(index_back=1, preserve_energy=100))
        # expect = np.array([[[[21.5, 22.0], [17.5, 13.]]]])
        expect = np.array([[[[21.75, 21.75], [18.75, 13.75]]]])
        np.testing.assert_array_almost_equal(
            x=expect, y=get_numpy(result),
            err_msg="The expected array x and computed y are not almost equal.")


