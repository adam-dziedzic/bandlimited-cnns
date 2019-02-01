import unittest
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.utils.log_utils import get_logger
import logging
import torch
import numpy as np
from cnns.nnlib.dct.reju_dct \
    import dC2e, dS2e, dct_convolution, fft_convolution, DST2e


class TestRejuDCT(unittest.TestCase):

    def setUp(self):
        print("\n")
        log_file = "test_reju_dct.log"
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

    def test_dC2e_odd_input_length(self):
        x = np.array([0.7577, 0.7431, 0.3922, 0.6555, 0.1712])
        expect = np.array([5.4395, -0.1458, -0.9044, 0.9044, 0.1458, 0])
        result = dC2e(x)
        np.testing.assert_allclose(actual=result, desired=expect,
                                   rtol=1e-3)

    def test_dC2e_even_input_length(self):
        x = np.array([0.5472, 0.1386, 0.1493, 0.2575, 0.8407, 0.2543])
        expect = np.array([4.3753, 0.6836, -0.7504, 0, 0.7504, -0.6836, 0])
        result = dC2e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_DS2e_odd_input_length(self):
        x = np.array([0.2551, 0.5060, 0.6991, 0.8909, 0.9593])
        expect = np.array([4.4089, -1.5600,  1.4301, -0.8869,  1.0332])
        result = DST2e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_DS2e_even_input_length(self):
        x = np.array([0.6541, 0.6892, 0.7482, 0.4505, 0.0838, 0.2290])
        expect = np.array([3.8660, 1.9335, 0.6469, 0.2208, 1.2332, 0.2346])
        result = DST2e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_dS2e_odd_input_length(self):
        x = np.array([0.2551, 0.5060, 0.6991, 0.8909, 0.9593])
        expect = np.array([0, -1.5600, -0.8869, -0.8869, -1.5600, 0])
        result = dS2e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_dct_convolution(self):
        s = np.array([0.6797, 0.6551, 0.1626, 0.1190, 0.4984])
        h = np.array([0.9597, 0.3404, 0.5853, 0.2238, 0.7513])
        expect = fft_convolution(s, h)
        # fft: [1.781406, 1.502568, 1.230928, 1.534484]
        # expect = np.array([1.0110, 1.2726, 1.4868, 1.5321, 1.0976])
        result = dct_convolution(s, h)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)
