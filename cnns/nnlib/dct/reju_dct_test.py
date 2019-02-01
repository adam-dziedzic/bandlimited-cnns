import unittest
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.utils.log_utils import get_logger
import logging
import torch
import numpy as np
from cnns.nnlib.dct.reju_dct import dC2e, dS2e, dct_convolution, \
    fft_convolution, rfft_convolution, DST1e, DST2e
from numpy.fft import rfft, irfft, fft, ifft

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
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_dC2e_even_input_length(self):
        x = np.array([0.5472, 0.1386, 0.1493, 0.2575, 0.8407, 0.2543])
        expect = np.array([4.3753, 0.6836, -0.7504, 0, 0.7504, -0.6836, 0])
        result = dC2e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_DS2e_odd_input_length(self):
        x = np.array([0.2551, 0.5060, 0.6991, 0.8909, 0.9593])
        expect = np.array([4.4089, -1.5600, 1.4301, -0.8869, 1.0332])
        result = DST2e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_DS2e_even_input_length(self):
        x = np.array([0.6541, 0.6892, 0.7482, 0.4505, 0.0838, 0.2290])
        expect = np.array([3.8660, 1.9335, 0.6469, 0.2208, 1.2332, 0.2346])
        result = DST2e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_DST1e_odd_input_length(self):
        x = np.array([0.2551, 0.5060, 0.6991, 0.8909, 0.9593])
        expect = np.array([5.0321, -1.8864, 1.0306, -0.5530, 0.1931])
        result = DST1e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_DST1e_even_input_length(self):
        x = np.array([0.6541, 0.6892, 0.7482, 0.4505, 0.0838, 0.2290])
        expect = np.array([4.3123, 2.1035, 0.5183, -0.1620, 0.9138, 0.0027])
        result = DST1e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-2)

    def test_dS2e_odd_input_length(self):
        x = np.array([0.2551, 0.5060, 0.6991, 0.8909, 0.9593])
        expect = np.array([0, -1.5600, -0.8869, -0.8869, -1.5600, 0])
        result = dS2e(x)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_dct_convolution_fft1(self):
        s = np.array([0.6797, 0.6551, 0.1626, 0.1190, 0.4984])
        h = np.array([0.9597, 0.3404, 0.5853, 0.2238, 0.7513])
        expect = rfft_convolution(s, h)
        print("expect: ", expect)
        expect2 = np.convolve(s, h, mode="same")
        print("expect2: ", expect2)
        expect3 = fft_convolution(s, h)
        print("expect3: ", expect3)
        # fft: [1.781406, 1.502568, 1.230928, 1.534484]
        # expect = np.array([1.0110, 1.2726, 1.4868, 1.5321, 1.0976])
        result = dct_convolution(s, h)
        print("result: ", result)
        np.testing.assert_allclose(actual=result, desired=expect3, rtol=1e-3)

    def test_dct_convolution_pure1(self):
        s = np.array([1, 2, 3, 4, 5.0], dtype=float)
        h = np.array([2, 3, -1, 0, -1.0], dtype=float)
        expect = np.array([11, -1.0, 7, 10, 18], dtype=float)
        print("expect: ", expect)
        expect2 = np.convolve(s, h, mode='same')
        print("expect2: ", expect2)
        expect3 = irfft(rfft(s) * rfft(h))
        print("expect3: ", expect3)
        expect4 = np.real(ifft(fft(s) * fft(h)))
        print("expect4: ", expect4)
        result = dct_convolution(s, h)
        print("result: ", result)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_dct_convolution_pure2(self):
        s = np.array([1, 2, -3, 4, 5.0, -2], dtype=float)
        h = np.array([2, 3, -1, 0, -1.0, 4], dtype=float)
        expect = np.array([2, -7, 10, 19, 16, 9.0], dtype=float)
        print("expect: ", expect)
        expect2 = np.real(ifft(fft(s) * fft(h)))
        print("expect2: ", expect2)
        result = dct_convolution(s, h)
        np.testing.assert_allclose(actual=result, desired=expect, rtol=1e-3)

    def test_dct_convolution_pure3(self):
        s = np.array([1, 2, 3, -4.0, 1.0], dtype=float)
        h = np.array([2, 3, -1, 0, -2.0], dtype=float)
        expect2 = np.convolve(s, h, mode='same')
        print("expect2: ", expect2)
        expect3 = irfft(rfft(s) * rfft(h))
        print("expect3: ", expect3)
        expect4 = np.real(ifft(fft(s) * fft(h)))
        print("expect4: ", expect4)
        result = dct_convolution(s, h)
        print("result: ", result)
        np.testing.assert_allclose(actual=result, desired=expect4, rtol=1e-3,
                                   atol=1e-10)

    def test_linear_convolution1(self):
        s = np.array([1, 2, 3, -4.0, 1.0], dtype=float)
        h = np.array([2, 3, -1], dtype=float)
        h_len = h.shape[-1]
        pad = h_len // 2 + 1
        s = np.pad(s, [0, pad], mode="constant")
        fft_len = s.shape[-1]
        h = np.pad(h, [0, fft_len - h_len], mode="constant")
        expect2 = np.convolve(s, h, mode='same')
        print("expect2: ", expect2)
        result = dct_convolution(s, h)
        print("result: ", result)
        np.testing.assert_allclose(actual=result, desired=expect2, rtol=1e-3,
                                   atol=1e-10)

if __name__ == '__main__':
    unittest.main()