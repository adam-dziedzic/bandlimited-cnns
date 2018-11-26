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


class TestBenchmarkConv2d(unittest.TestCase):

    def setUp(self):
        log_file = "conv2D_benchmark.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def test_mem_usage(self):
        dtype=torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device used: ", str(device))
        N, C, H, W = 128, 128, 32, 32
        K, HH, WW = 128, 3, 3
        x = torch.randn(N, C, H, W, dtype=dtype, device=device)
        y = torch.randn(K, C, HH, WW, dtype=dtype, device=device)

        start = time.time()
        convStandard = torch.nn.functional.conv2d(input=x, weight=y, stride=2)
        print("convStandard time: ", time.time() - start)

        conv = Conv2dfftFunction()
        start = time.time()
        convFFT = conv.forward(ctx=None, input=x, filter=y, stride=2,
                              args=Arguments(stride_type=StrideType.STANDARD))
        print("convFFT time: ", time.time() - start)

        np.testing.assert_array_almost_equal(
            x=convStandard.cpu().detach().numpy(),
            y=convFFT.cpu().detach().numpy(), decimal=4,
            err_msg="The expected array x and computed y are not almost equal.")