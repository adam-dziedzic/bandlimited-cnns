from cnns.nnlib.pytorch_layers.fft_band_2D_complex_mask import \
    FFTBandFunctionComplexMask2D

import unittest
import torch
from cnns.nnlib.pytorch_layers.fft_band_2D import FFTBand2D
from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.datasets.cifar10_example import cifar10_example
import numpy as np


class TestFFTBandFunction2DdiskMask(unittest.TestCase):

    def test_fft_band_function_2D_disk_mask(self):
        args = Arguments()
        args.dtype = torch.float
        args.device = torch.device("cpu")
        args.values_per_channel = 2
        args.compress_rate = 0.4

        input = np.arange(0, 49).reshape((1, 1, 7, 7))
        a = torch.tensor(input, requires_grad=True,
                         dtype=args.dtype, device=args.device)

        result = FFTBandFunctionComplexMask2D.forward(ctx=None, input=a,
                                                      compress_rate=26, val=0)
        print("result: ", result)
