from cnns.nnlib.utils.complex_mask import get_disk_mask
from cnns.nnlib.utils.complex_mask import get_hyper_mask
from cnns.nnlib.utils.complex_mask import get_inverse_hyper_mask

import torch
import unittest
import numpy as np


class TestGetComplexMask(unittest.TestCase):

    def test_get_complex_mask(self):
        mask, array_mask = get_disk_mask(H=7, W=7, compress_rate=26, val=0)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 0., 1., 1., 1.],
             [1., 1., 0., 0., 0., 1., 1.],
             [1., 0., 0., 0., 0., 0., 1.],
             [1., 1., 0., 0., 0., 1., 1.],
             [1., 1., 1., 0., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1.]])
        np.testing.assert_equal(actual=array_mask, desired=desired_array_mask)

        desired = torch.tensor([[[1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.]],

                                [[1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [0., 0.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.]],

                                [[1., 1.],
                                 [1., 1.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [1., 1.],
                                 [1., 1.]],

                                [[1., 1.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [1., 1.]],

                                [[1., 1.],
                                 [1., 1.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [1., 1.],
                                 [1., 1.]],

                                [[1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [0., 0.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.]],

                                [[1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.],
                                 [1., 1.]]])
        np.testing.assert_equal(actual=mask.numpy(), desired=desired.numpy())


    def test_get_complex_mask_linear(self):
        mask, array_mask = get_disk_mask(H=7, W=7, compress_rate=26, val=0,
                                         interpolate="linear")
        array_mask = torch.tensor(array_mask)
        print("array mask:\n", array_mask)
        # print("mask:\n", array_mask)
        desired_array_mask = np.array(
            [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.6713, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 0.6713, 0.3379, 0.6713, 1.0000, 1.0000],
             [1.0000, 0.6713, 0.3379, 0.0046, 0.3379, 0.6713, 1.0000],
             [1.0000, 1.0000, 0.6713, 0.3379, 0.6713, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.6713, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
        np.testing.assert_allclose(actual=array_mask.numpy(),
                                   desired=desired_array_mask, rtol=1e-02)

    def test_get_complex_mask_exponent(self):
        mask, array_mask = get_disk_mask(H=7, W=7, compress_rate=26, val=0,
                                         interpolate="exponent")
        array_mask = torch.tensor(array_mask)
        print("array mask:\n", )
        # print("mask:\n", array_mask)
        desired_array_mask = np.array(
            [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.3730, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 0.3730, 0.1372, 0.3730, 1.0000, 1.0000],
             [1.0000, 0.3730, 0.1372, 0.0505, 0.1372, 0.3730, 1.0000],
             [1.0000, 1.0000, 0.3730, 0.1372, 0.3730, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.3730, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
        np.testing.assert_allclose(actual=array_mask.numpy(),
                                   desired=desired_array_mask, rtol=1e-03)

    def test_get_complex_mask_log(self):
        mask, array_mask = get_disk_mask(H=7, W=7, compress_rate=26,
                                         val=0,
                                         interpolate="log")
        array_mask = torch.tensor(array_mask)
        print("array mask:\n")
        print("mask:\n", array_mask)
        desired_array_mask = np.array(
            [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.7671, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 0.7671, 0.4578, 0.7671, 1.0000, 1.0000],
             [1.0000, 0.7671, 0.4578, 0.0079, 0.4578, 0.7671, 1.0000],
             [1.0000, 1.0000, 0.7671, 0.4578, 0.7671, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 0.7671, 1.0000, 1.0000, 1.0000],
             [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]
        )
        np.testing.assert_allclose(actual=array_mask.numpy(),
                                   desired=desired_array_mask, rtol=1e-02)

    def test_get_hyper_mask1(self):
        H = 7
        W = 7
        compress_rate = 26
        mask, array_mask = get_hyper_mask(H=H, W=W,
                                          compress_rate=compress_rate, val=0,
                                          onesided=False)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1., 1., 1., 0., 1., 1., 1.],
             [1., 1., 1., 0., 1., 1., 1.],
             [1., 1., 1., 0., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 1., 1., 1.],
             [1., 1., 1., 0., 1., 1., 1.],
             [1., 1., 1., 0., 1., 1., 1.]])
        np.testing.assert_equal(actual=array_mask, desired=desired_array_mask)
        self.zeros_test(compress_rate=compress_rate, array_mask=array_mask,
                        H=H, W=W)

    def test_get_hyper_mask2(self):
        compress_rate = 40
        H = 10
        W = 10
        mask, array_mask = get_hyper_mask(H=H, W=W,
                                          compress_rate=compress_rate, val=0,
                                          onesided=False)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
             [1., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
             [1., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
             [1., 1., 1., 0., 0., 0., 0., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 0., 1., 1., 1.],
             [1., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
             [1., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
             [1., 1., 1., 1., 0., 0., 1., 1., 1., 1.]]
        )
        np.testing.assert_equal(actual=array_mask, desired=desired_array_mask)
        self.zeros_test(compress_rate=compress_rate, array_mask=array_mask,
                        H=H, W=W)

    def test_get_hyper_mask3(self):
        H, W, compress_rate = 7, 7, 50
        mask, array_mask = get_hyper_mask(H=H, W=W,
                                          compress_rate=compress_rate, val=0,
                                          onesided=False)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1., 1., 0., 0., 0., 1., 1.],
             [1., 1., 0., 0., 0., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 0., 0., 0., 1., 1.],
             [1., 1., 0., 0., 0., 1., 1.]])
        np.testing.assert_equal(actual=array_mask, desired=desired_array_mask)
        self.zeros_test(compress_rate=compress_rate, array_mask=array_mask,
                        H=H, W=W)

    def test_get_inverse_hyper_mask3(self):
        H, W, compress_rate = 7, 7, 50
        mask, array_mask = get_inverse_hyper_mask(H=H, W=W,
                                          compress_rate=compress_rate, val=0,
                                          onesided=False)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[0., 0., 1., 1., 1., 0., 0.],
             [0., 0., 1., 1., 1., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 1., 1., 1., 0., 0.],
             [0., 0., 1., 1., 1., 0., 0.]])
        np.testing.assert_equal(actual=array_mask, desired=desired_array_mask)
        self.zeros_test(compress_rate=compress_rate, array_mask=array_mask,
                        H=H, W=W)

    def test_get_hyper_mask4(self):
        H = 7
        W = 7
        compress_rate = 80
        mask, array_mask = get_hyper_mask(H=H, W=W,
                                          compress_rate=compress_rate, val=0,
                                          onesided=False)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1., 0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0., 1.]])
        np.testing.assert_equal(actual=array_mask, desired=desired_array_mask)
        self.zeros_test(compress_rate=compress_rate, array_mask=array_mask,
                        H=H, W=W)

    def zeros_test(self, compress_rate, array_mask, H, W):
        compress_rate = compress_rate / 100
        print("array mask:\n", array_mask)
        zeros_size = np.sum(array_mask == 0.0)
        # print(zero2)
        total_size = H * W
        # print("total size: ", total_size)
        fraction_zeroed = zeros_size / total_size
        print("compress rate: ", compress_rate, " fraction of zeroed out: ",
              fraction_zeroed)
        error = 0.2
        if fraction_zeroed > compress_rate + error or (
                fraction_zeroed < compress_rate - error):
            raise Exception(f"The compression is wrong, for compression "
                            f"rate {compress_rate}, the number of fraction "
                            f"of zeroed out coefficients "
                            f"is: {fraction_zeroed}")

    def test_get_hyper_mask_lin(self):
        mask, array_mask = get_hyper_mask(H=7, W=7, compress_rate=80, val=0,
                                          interpolate="linear",
                                          onesided=False)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1.0000, 0.8000, 0.6000, 0.4000, 0.6000, 0.8000, 1.0000],
             [0.8000, 0.8000, 0.6000, 0.4000, 0.6000, 0.8000, 0.8000],
             [0.6000, 0.6000, 0.6000, 0.4000, 0.6000, 0.6000, 0.6000],
             [0.4000, 0.4000, 0.4000, 0.2000, 0.4000, 0.4000, 0.4000],
             [0.6000, 0.6000, 0.6000, 0.4000, 0.6000, 0.6000, 0.6000],
             [0.8000, 0.8000, 0.6000, 0.4000, 0.6000, 0.8000, 0.8000],
             [1.0000, 0.8000, 0.6000, 0.4000, 0.6000, 0.8000, 1.0000]]
        )
        np.testing.assert_equal(actual=array_mask, desired=desired_array_mask)

    def test_get_hyper_mask_exp(self):
        mask, array_mask = get_hyper_mask(H=7, W=7, compress_rate=80, val=0,
                                          interpolate="exp",
                                          onesided=False)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1.0000, 0.3679, 0.1353, 0.0498, 0.1353, 0.3679, 1.0000],
             [0.3679, 0.3679, 0.1353, 0.0498, 0.1353, 0.3679, 0.3679],
             [0.1353, 0.1353, 0.1353, 0.0498, 0.1353, 0.1353, 0.1353],
             [0.0498, 0.0498, 0.0498, 0.0183, 0.0498, 0.0498, 0.0498],
             [0.1353, 0.1353, 0.1353, 0.0498, 0.1353, 0.1353, 0.1353],
             [0.3679, 0.3679, 0.1353, 0.0498, 0.1353, 0.3679, 0.3679],
             [1.0000, 0.3679, 0.1353, 0.0498, 0.1353, 0.3679, 1.0000]]
        )
        np.testing.assert_allclose(actual=array_mask, desired=desired_array_mask,
                                rtol=1e-3)

    def test_get_hyper_mask_log(self):
        mask, array_mask = get_hyper_mask(H=7, W=7, compress_rate=80, val=0,
                                          interpolate="log",
                                          onesided=False)
        print("array mask:\n", torch.tensor(array_mask))
        print("mask:\n", mask)
        desired_array_mask = np.array(
            [[1.0000, 0.8648, 0.7085, 0.5231, 0.7085, 0.8648, 1.0000],
             [0.8648, 0.8648, 0.7085, 0.5231, 0.7085, 0.8648, 0.8648],
             [0.7085, 0.7085, 0.7085, 0.5231, 0.7085, 0.7085, 0.7085],
             [0.5231, 0.5231, 0.5231, 0.2954, 0.5231, 0.5231, 0.5231],
             [0.7085, 0.7085, 0.7085, 0.5231, 0.7085, 0.7085, 0.7085],
             [0.8648, 0.8648, 0.7085, 0.5231, 0.7085, 0.8648, 0.8648],
             [1.0000, 0.8648, 0.7085, 0.5231, 0.7085, 0.8648, 1.0000]]
        )
        np.testing.assert_allclose(actual=array_mask, desired=desired_array_mask,
                                rtol=1e-3)



if __name__ == '__main__':
    unittest.main()
