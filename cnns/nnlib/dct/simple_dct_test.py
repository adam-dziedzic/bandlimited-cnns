import unittest
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.utils.log_utils import get_logger
import logging
import torch
import numpy as np
from cnns.nnlib.dct.simple_dct import DCT


class TestSimpleDCT(unittest.TestCase):

    def setUp(self):
        print("\n")
        log_file = "test_simple_dct.log"
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

    def testCorrelation1(self):
        x = np.array([1, 2, 3], dtype=float)
        y = np.array([-1, 2], dtype=float)
        dct = DCT()
        expect = np.correlate(x, y, mode='full')[len(y) - 1:]
        print("expect: ", expect)
        result = dct.correlate_izumi(x, y, use_next_power2=False)
        print("result: ", result)
        assert np.testing.assert_allclose(actual=result, desired=expect)

    def testCorrelation2(self):
        x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8.0], dtype=float)
        y = np.array([1.0, 4.0, 1.0], dtype=float)
        dct = DCT()
        expect = np.correlate(x, y, mode='full')[len(y) - 1:]
        print("expect: ", expect)
        result = dct.correlate_izumi(x, y, use_next_power2=False)
        print("result: ", result)
        assert np.testing.assert_allclose(actual=result, desired=expect)

    def testCorrelation3(self):
        x = np.array([1.0, 2, 3, 4, 4, 3, 2, 1.0], dtype=float)
        y = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=float)
        dct = DCT()
        expect = np.correlate(x, y, mode='full')
        print("expect: ", expect)
        result = dct.correlate_izumi(x, y, use_next_power2=False)
        print("result: ", result)
        assert np.testing.assert_allclose(actual=result, desired=expect)

    def test_dct2(self):
        dct = DCT()
        x = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0], dtype=float)
        xdct = dct.dct2(x)
        print("xdct: ", xdct)

        x2 = np.array([1.0,2.0,3.0,4.0,4.0,3.0,2.0,1.0], dtype=float)
        xdct2 = dct.dct2(x2)
        print("xdct2 even and symmetric: ", xdct2)

        x3 = np.array([1.0,2.0,3.0,4.0,3.0,2.0,1.0], dtype=float)
        xdct3 = dct.dct2(x3)
        print("xdct3 even and symmetric: ", xdct3)

    def testConvolution1(self):
        x = np.array([1.0, 2, 3, 4, 4, 3, 2, 1.0], dtype=float)
        y = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=float)
        dct = DCT()
        expect = np.convolve(x, y, mode='full')
        print("expect: ", expect)
        result = dct.correlate_izumi(x, y, use_next_power2=False,
                                     is_convolution=True)
        print("result: ", result)
        assert np.testing.assert_allclose(actual=result, desired=expect)

    def testCorrelation2(self):
        x = np.array([1.0, -2.0, 3.0, -4.0, 2.0, -8.0, -1.0, 5.0], dtype=float)
        y = np.array([1.0, -4.0, 2.0], dtype=float)
        dct = DCT()
        expect = np.correlate(x, y, mode='full')[len(y) - 1:]
        print("expect: ", expect)
        result = dct.correlate_izumi(x, y, use_next_power2=False)
        print("result: ", result)
        assert np.testing.assert_allclose(actual=result, desired=expect)

