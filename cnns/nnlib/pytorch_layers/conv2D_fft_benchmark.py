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
from cnns.nnlib.utils.arguments import Arguments

"""
Results:
Testing started at 10:03 PM ...
ssh://ady@skr-compute1:22/home/ady/anaconda3/bin/python3.6 -u /home/ady/.pycharm_helpers/pycharm/_jb_unittest_runner.py --target conv2D_fft_benchmark.TestBenchmarkConv2d.test_mem_usage
Launching unittests with arguments python -m unittest conv2D_fft_benchmark.TestBenchmarkConv2d.test_mem_usage in /local/code/time-series-ml/cnns/nnlib/pytorch_layers
2018-11-27 22:03:53,779 - root - INFO - set_up_logging(19)- started logging to: conv2D_benchmark.log
2018-11-27 22:03:53,779 - conv2D_fft_benchmark - INFO - setUp(31)- Set up test
device used:  cuda
convStandard time:  0.0043790340423583984
fft_forward_time:  0.00822305679321289
backward pass with step
create tensor time:  0.00011348724365234375
correlation time:  0.0046923160552978516
irfft time:  0.012486696243286133
convFFT time:  0.028664827346801758
Pytorch speedup is: 6.545924756356509 X


Ran 1 test in 4.760s

OK
Process finished with exit code 0

Testing started at 11:23 PM ...
ssh://ady@skr-compute1:22/home/ady/anaconda3/bin/python3.6 -u /home/ady/.pycharm_helpers/pycharm/_jb_unittest_runner.py --target conv2D_fft_benchmark.TestBenchmarkConv2d.test_forward_backward
Launching unittests with arguments python -m unittest conv2D_fft_benchmark.TestBenchmarkConv2d.test_forward_backward in /local/code/time-series-ml/cnns/nnlib/pytorch_layers
2018-11-27 23:24:01,603 - root - INFO - set_up_logging(19)- started logging to: conv2D_benchmark.log
2018-11-27 23:24:01,603 - conv2D_fft_benchmark - INFO - setUp(55)- Set up test
device used:  cuda
convStandard time:  0.004107236862182617
fft_forward_time:  0.008202791213989258
backward pass with step
create tensor time:  0.00011038780212402344
correlation time:  0.004713535308837891
restore time:  4.76837158203125e-06
irfft time:  0.012326240539550781
convFFT time:  0.02847886085510254
Pytorch forward pass speedup is: 6.9338248098914494 X
standard back time:  0.010608196258544922
total multiply time:  0.00040841102600097656
total restore time:  5.9604644775390625e-06
total multiply time:  0.0008776187896728516
total restore time:  1.3589859008789062e-05
conv fft back time:  0.4154634475708008
Pytorch speedup for backprop: 39.164381714388455 X
"""


class TestBenchmarkConv2d(unittest.TestCase):

    def setUp(self):
        log_file = "conv2D_benchmark.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def test_forward(self):
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device used: ", str(device))
        N, C, H, W = 128, 16, 32, 32
        K, HH, WW = 16, 3, 3
        x = torch.randn(N, C, H, W, dtype=dtype, device=device)
        y = torch.randn(K, C, HH, WW, dtype=dtype, device=device)

        start = time.time()
        convStandard = torch.nn.functional.conv2d(input=x, weight=y, stride=1)
        convStandardTime = time.time() - start
        print("convStandard time: ", convStandardTime)

        conv = Conv2dfftFunction()
        start = time.time()
        convFFT = conv.forward(ctx=None, input=x, filter=y, stride=1,
                               args=Arguments(stride_type=StrideType.STANDARD))
        convFFTtime = time.time() - start
        print("convFFT time: ", convFFTtime)
        speedup = convFFTtime / convStandardTime
        print(f"Pytorch speedup is: {speedup} X")

        np.testing.assert_array_almost_equal(
            x=convStandard.cpu().detach().numpy(),
            y=convFFT.cpu().detach().numpy(), decimal=3,
            err_msg="The expected array x and computed y are not almost equal.")

    def test_forward_compression(self):
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device used: ", str(device))
        N, C, H, W = 128, 16, 32, 32
        K, HH, WW = 16, 3, 3
        x = torch.randn(N, C, H, W, dtype=dtype, device=device)
        y = torch.randn(K, C, HH, WW, dtype=dtype, device=device)

        start = time.time()
        torch.nn.functional.conv2d(input=x, weight=y, stride=1)
        convStandardTime = time.time() - start
        print("convStandard time: ", convStandardTime)

        conv = Conv2dfftFunction()
        start = time.time()
        conv.forward(ctx=None, input=x, filter=y, stride=1,
                     args=Arguments(stride_type=StrideType.STANDARD,
                                    preserve_energy=90))
        convFFTtime = time.time() - start
        print("convFFT time: ", convFFTtime)
        speedup = convFFTtime / convStandardTime
        print(f"Pytorch speedup is: {speedup} X")

    def test_forward_backward(self):
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device used: ", str(device))
        N, C, H, W = 128, 16, 32, 32
        K, HH, WW = 16, 3, 3
        x = torch.randn(N, C, H, W, dtype=dtype, device=device,
                        requires_grad=True)
        x_expect = x.clone().detach().requires_grad_(True)
        y = torch.randn(K, C, HH, WW, dtype=dtype, device=device,
                        requires_grad=True)
        y_expect = y.clone().detach().requires_grad_(True)
        start = time.time()
        convStandard = torch.nn.functional.conv2d(input=x_expect,
                                                  weight=y_expect, stride=1)
        convStandardTime = time.time() - start
        print("convStandard time: ", convStandardTime)

        conv = Conv2dfft(filter_value=y, stride=1, bias=False,
                         args=Arguments(stride_type=StrideType.STANDARD))
        start = time.time()
        convFFT = conv.forward(input=x)
        convFFTtime = time.time() - start
        print("convFFT time: ", convFFTtime)
        speedup = convFFTtime / convStandardTime
        print(f"Pytorch forward pass speedup is: {speedup} X")

        np.testing.assert_array_almost_equal(
            x=convStandard.cpu().detach().numpy(),
            y=convFFT.cpu().detach().numpy(), decimal=3,
            err_msg="The expected array x and computed y are not almost equal.")

        dout = torch.randn(list(convStandard.size()), device=device,
                           dtype=dtype)
        dout_clone = dout.clone()

        standard_back_time_start = time.time()
        convStandard.backward(dout)
        standard_back_time = time.time() - standard_back_time_start
        print("standard back time: ", standard_back_time)

        fft_back_time_start = time.time()
        convFFT.backward(dout_clone)
        conv_fft_back_time = time.time() - fft_back_time_start
        assert conv.is_manual[0] == 1
        print("conv fft back time: ", conv_fft_back_time)
        speedup = conv_fft_back_time / standard_back_time
        print(f"Pytorch speedup for backprop: {speedup} X")

        np.testing.assert_array_almost_equal(x.grad.cpu().detach().numpy(),
                                             x_expect.grad.cpu().detach().numpy(),
                                             decimal=3)

        np.testing.assert_array_almost_equal(y.grad.cpu().detach().numpy(),
                                             y_expect.grad.cpu().detach().numpy(),
                                             decimal=3)




