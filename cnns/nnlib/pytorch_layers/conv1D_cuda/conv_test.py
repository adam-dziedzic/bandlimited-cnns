import unittest
import logging
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
import numpy as np
import torch
from cnns.nnlib.pytorch_layers.conv1D_cuda.conv import Conv1dfftCuda

import conv1D_cuda


class TestPyTorchConv1d(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv1D_cuda_reuse_map_fft.log"
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
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")
        if torch.cuda.is_available() is False:
            self.fail("This test can be executed only on GPU!")
        device = torch.device("lltm_cuda")
        print("Conv Cuda")
        conv = Conv1dfftCuda(filter_value=torch.tensor(y, device=device),
                             bias_value=torch.tensor(b, device=device))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))

    def test_plus_reduce(self):
        x = torch.tensor([1, 2, 3, 4])
        result = conv1D_cuda.plus_reduce(x)
        np.testing.assert_almost_equal(result, torch.tensor(10))
