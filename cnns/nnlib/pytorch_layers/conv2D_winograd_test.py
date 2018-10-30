from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
import logging
import unittest
import numpy as np
import torch


class TestPyTorchConv1d(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv1D_reuse_map_fft.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def testSimpleWinograd(self):
        x = np.array([[[1., 2., 3., 1., 4., 5.]]])
        y = np.array([[[2., 1., 3.]]])
        b = np.array([0.0])
        # get the expected results from numpy correlate
        expected_result = np.correlate(x[0, 0, :], y[0, 0, :], mode="valid")

        conv = Conv1dfftCompressSignalOnly(filter_value=torch.from_numpy(y),
                                           bias_value=torch.from_numpy(b))
        result = conv.forward(input=torch.from_numpy(x))
        np.testing.assert_array_almost_equal(
            result, np.array([[expected_result]]))