import torch
import unittest
import numpy as np

from cnns.nnlib.pytorch_layers.fft_band_2D import FFTBand2D
from cnns.nnlib.pytorch_layers.fft_band_2D import FFTBandFunction2D
from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.datasets.cifar10_example import cifar10_example
from cnns.nnlib.pytorch_layers.pytorch_utils import MockContext


class TestFFTBand2D(unittest.TestCase):

    def test_FFTBand2D(self):
        # uncomment the line below to run on gpu
        # device = torch.device("conv1D_cuda:0")

        # To apply our Function, we use Function.apply method.
        # We alias this as 'relu'.
        args = Arguments()
        args.dtype = torch.float
        args.device = torch.device("cpu")
        args.values_per_channel = 2
        args.compress_rate = 0.4

        C, H, W = 3, 32, 32
        # input = np.random.rand(1, 1, 32, 32)
        input = [cifar10_example]
        a = torch.tensor(input, requires_grad=True,
                         dtype=args.dtype, device=args.device)
        print("a grad: ", a.grad)

        print('L2 distance between the input image and compressed image:')
        print("compress rate, zeroed %, L2 distance")
        ctx = MockContext()
        for compress_rate in [x * 10 for x in range(0, 11, 1)]:
            args.compress_rate = compress_rate
            result = FFTBandFunction2D.forward(
                ctx=ctx,
                input=a,
                onesided=True,
                compress_rate=args.compress_rate,
                is_test=True)
            print(args.compress_rate, ",", ctx.fraction_zeroed * 100,
                  torch.dist(a, result, 2).item())

        args.compress_rate = 0.1
        band = FFTBand2D(args)
        result = band(a)
        print(
            f"L2 distance from origin for compress rate {args.compress_rate}: ",
            torch.dist(a, result, 2).item())

        gradient = np.arange(C * H * W).reshape(1, C, H, W) / 10
        gradienty_y = torch.tensor(gradient, dtype=args.dtype,
                                   device=args.device)
        result.backward(gradienty_y)
        # print("data a: ", a)
        # print("a final grad: ", a.grad)
        expected_gradient = torch.tensor(gradient, dtype=args.dtype,
                                         device=args.device)
        self.assertTrue(a.grad.equal(expected_gradient))

    def test_zero_out(self):
        compress_rate = 0.5
        print("compress rate: ", compress_rate)
        torch.set_printoptions(threshold=5000)
        xfft = np.arange(32 * 32).reshape(32, 32)
        xfft = torch.tensor(xfft)
        # print("xfft1:", xfft)
        H_xfft, W_xfft = xfft.size()
        H_compress = np.sqrt(1.0 - compress_rate) * H_xfft
        W_compress = int(np.sqrt(1.0 - compress_rate) * W_xfft)
        H_top = int(H_compress // 2 + 1)
        H_bottom = H_compress // 2
        H_end = int(H_xfft - H_bottom)

        # zero out high energy coefficients
        zero1 = torch.sum(xfft == 0.0).item()
        print()
        print("zero1: ", zero1)
        xfft[H_top:H_end, :] = 0.0
        xfft[:, W_compress:] = 0.0
        # print("xfft1:", xfft)
        zero2 = torch.sum(xfft == 0.0).item()
        print("zero2: ", zero2)
        total_size = H_xfft * W_xfft
        print("total size: ", total_size)
        print("fraction of zeroed out: ", (zero2 - zero1) / total_size)

    def test_zero_out2(self):
        compress_rate = 0.5
        print("compress rate: ", compress_rate)
        torch.set_printoptions(threshold=5000)
        input = torch.rand((1, 1, 7, 7))
        ctx = MockContext()
        out = FFTBandFunction2D.forward(ctx=ctx, input=input,
                                        compress_rate=compress_rate,
                                        onesided=True,
                                        is_next_power2=False)


if __name__ == '__main__':
    unittest.main()
