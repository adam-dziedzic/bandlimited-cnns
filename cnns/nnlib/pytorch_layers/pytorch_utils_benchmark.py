import torch
import unittest
from cnns.nnlib.pytorch_layers.pytorch_utils import flip
from cnns.nnlib.pytorch_layers.pytorch_utils import preserve_energy2D
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul
from cnns.nnlib.pytorch_layers.conv2D_fft import Conv2dfft
from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import ConvExecType
import numpy as np
from torch import tensor
import socket
import time

if torch.cuda.is_available():
    from complex_mul_cuda import \
        complex_mul_stride_no_permute as complex_mul_stride_no_permute_cuda
    from complex_mul_cuda import \
        complex_mul_shared_log as complex_mul_shared_log_cuda


class TestPytorchUtils(unittest.TestCase):

    def test_cuda_stride_no_permute_multiply_big2(self):
        N, C, H, W, I = 32, 128, 8, 4, 2
        F = 256
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        dtype = torch.float
        x = torch.randn(N, C, H, W, I, device=device, dtype=dtype)
        y = torch.randn(F, C, H, W, I, device=device, dtype=dtype)
        out = torch.zeros(N, F, H, W, I, device=device, dtype=dtype)

        start = time.time()
        complex_mul_stride_no_permute_cuda(x, y, out, 1024)
        cuda_mul_time = time.time() - start
        print("\ncuda mul time: ", cuda_mul_time)

        x = x.unsqueeze(dim=1)
        start = time.time()
        expect = complex_mul(x, y)
        expect = expect.sum(dim=2)
        pytorch_mul_time = time.time() - start
        print("pytorch mul time: ", pytorch_mul_time)

        print(f"pytorch is faster: {cuda_mul_time/pytorch_mul_time} X times")

        print("out.size(): ", out.size())

        np.testing.assert_allclose(
            actual=out.cpu().numpy(), desired=expect.cpu().numpy(), rtol=1e-4,
            err_msg="actual out different from desired expected")

    def test_cuda_stride_no_permute_multiply_big2_repeat(self):
        if not torch.cuda.is_available():
            print("CUDA is not available")

        N, C, H, W, I = 32, 128, 8, 4, 2
        F = 256
        repeat = 1000

        device = torch.device("cuda")
        dtype = torch.float
        x = torch.randn(N, C, H, W, I, device=device, dtype=dtype)
        y = torch.randn(F, C, H, W, I, device=device, dtype=dtype)

        start = time.time()

        for _ in range(repeat):
            out = torch.zeros(N, F, H, W, I, device=device, dtype=dtype)
            complex_mul_stride_no_permute_cuda(x, y, out, 1024)

        cuda_mul_time = time.time() - start
        print("\ncuda mul time: ", cuda_mul_time)

        x = x.unsqueeze(dim=1)
        start = time.time()

        for _ in range(repeat):
            expect = complex_mul(x, y)
            expect = expect.sum(dim=2)

        pytorch_mul_time = time.time() - start
        print("pytorch mul time: ", pytorch_mul_time)

        print(f"pytorch is faster: {cuda_mul_time/pytorch_mul_time} X times")

        print("out.size(): ", out.size())

        np.testing.assert_allclose(
            actual=out.cpu().numpy(), desired=expect.cpu().numpy(), rtol=1e-4,
            err_msg="actual out different from desired expected")

    def test_cuda_shared_log_multiply_big2(self):
        if not torch.cuda.is_available():
            print("No cuda device is available!")
        device = torch.device("cuda")
        dtype = torch.float

        repeat = 1

        N = 32
        I = 2
        # cases: F, C, H, W
        cases = [(64, 3, 32, 32),
                 (128, 64, 8, 5),
                 (64, 3, 119, 60),
                 (64, 64, 59, 30),
                 (128, 64, 57, 29),
                 (128, 128, 31, 16),
                 (256, 128, 31, 16),
                 (256, 256, 16, 9),
                 (512, 256, 16, 9),
                 (512, 512, 8, 5)]

        for case in cases:
            print("\ncase: ", case)
            F, C, H, W = case

            x = torch.randn(N, C, H, W, I, device=device, dtype=dtype)
            y = torch.randn(F, C, H, W, I, device=device, dtype=dtype)
            out = torch.zeros(N, F, H, W, I, device=device, dtype=dtype)

            # warm-up
            z = x*x
            del z
            w = y*y
            del w

            start = time.time()
            for _ in range(repeat):
                out = torch.irfft(input=x,
                                  signal_ndim=2,
                                  onesided=True)
            pytorch_fft_time = time.time() - start
            print("pytorch fft time: ", pytorch_fft_time)


            start = time.time()
            for _ in range(repeat):
                complex_mul_stride_no_permute_cuda(x, y, out, 1024)
            print("cuda stride no permute mul time: ", time.time() - start)

            start = time.time()
            for _ in range(repeat):
                # Move the channels to the last but one dimension.
                # We want for xfft: N, H, W, C, I.
                x_clone = x.permute(0, 2, 3, 1, 4).contiguous()
                # We want for yfft: F, H, W, C, I.
                y_clone = y.permute(0, 2, 3, 1, 4).contiguous()
                complex_mul_shared_log_cuda(x_clone, y_clone, out)
            print("cuda shared log mul time: ", time.time() - start)

            x = x.unsqueeze(dim=1)
            start = time.time()
            for _ in range(repeat):
                expect = complex_mul(x, y)
                expect = expect.sum(dim=2)
            print("pytorch mul time: ", time.time() - start)

            # print("out.size(): ", out.size())

            # np.testing.assert_allclose(
            #     actual=out.cpu().numpy(), desired=expect.cpu().numpy(), rtol=2,
            #     err_msg="actual out different from desired expected")

    def test_cuda_stride_no_permute_multiply_big2_repeat_sync(self):
        if not torch.cuda.is_available():
            print("CUDA is not available")
        N = 32
        I = 2
        # C, H, W, F, HH, WW = 3, 119, 60, 2, 64, 32, 32
        # C, H, W, F, HH, WW = 3, 128, 65, 64, 32, 32
        # C, H, W, F, HH, WW = 32, 512, 8, 5, 2, 512, 4, 4
        C, H, W, F, HH, WW = 512, 8, 5, 512, 2, 2


        repeat = 100

        device = torch.device("cuda")
        dtype = torch.float
        x = torch.randn(N, C, H, W, I, device=device, dtype=dtype)
        y = torch.randn(F, C, H, W, I, device=device, dtype=dtype)

        print("\ntesting\n")

        start = time.time()
        for _ in range(repeat):
            torch.cuda.synchronize()
        pytorch_sync_time = time.time() - start
        print("pytorch sync time: ", pytorch_sync_time)

        x_conv = torch.randn(N, C, HH, WW, device=device, dtype=dtype,
                             requires_grad=True)
        y_conv = torch.randn(F, C, HH, WW, device=device, dtype=dtype,
                             requires_grad=True)
        start = time.time()
        for _ in range(repeat):
            convStandard = torch.nn.functional.conv2d(
                input=x_conv, weight=y_conv, stride=1, padding=1)
            torch.cuda.synchronize()
        convStandardTime = time.time() - start
        print("convStandard time: ", convStandardTime)

        conv = Conv2dfft(weight_value=y_conv, stride=1, bias=False, padding=1,
                         args=Arguments(stride_type=StrideType.STANDARD,
                                        conv_exec_type=ConvExecType.CUDA,
                                        preserve_energy=100))
        start = time.time()
        for _ in range(repeat):
            convFFT = conv.forward(input=x_conv)
            torch.cuda.synchronize()
        convFFTtime = time.time() - start
        print("convFFT time: ", convFFTtime)

        speedup = convFFTtime / convStandardTime
        print(f"Pytorch forward pass speedup is: {speedup} X")

        # warm-up
        out = torch.irfft(input=x,
                          signal_ndim=2,
                          onesided=True)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(repeat):
            out = torch.irfft(input=x,
                              signal_ndim=2,
                              onesided=True)
            torch.cuda.synchronize()
        pytorch_fft_time = time.time() - start
        print("pytorch fft time: ", pytorch_fft_time)

        # warm-up
        out = torch.zeros(N, F, H, W, I, device=device, dtype=dtype)
        complex_mul_stride_no_permute_cuda(x, y, out, H*W)
        torch.cuda.synchronize()

        start = time.time()

        for _ in range(repeat):
            out = torch.zeros(N, F, H, W, I, device=device, dtype=dtype)
            complex_mul_stride_no_permute_cuda(x, y, out, 1024)
            torch.cuda.synchronize()
        cuda_mul_time = time.time() - start
        print("cuda stride no permute mul time: ", cuda_mul_time)

        # warm-up
        out = torch.zeros(N, F, H, W, I, device=device, dtype=dtype)
        # Move the channels to the last but one dimension.
        # We want for xfft: N, H, W, C, I.
        x_clone = x.permute(0, 2, 3, 1, 4).contiguous()
        # We want for yfft: F, H, W, C, I.
        y_clone = y.permute(0, 2, 3, 1, 4).contiguous()
        complex_mul_shared_log_cuda(x_clone, y_clone, out)
        torch.cuda.synchronize()

        start = time.time()

        # Move the channels to the last but one dimension.
        # We want for xfft: N, H, W, C, I.
        x_clone = x.permute(0, 2, 3, 1, 4).contiguous()
        # We want for yfft: F, H, W, C, I.
        y_clone = y.permute(0, 2, 3, 1, 4).contiguous()
        for _ in range(repeat):
            out = torch.zeros(N, F, H, W, I, device=device, dtype=dtype)
            complex_mul_shared_log_cuda(x_clone, y_clone, out)
            torch.cuda.synchronize()
        cuda_mul_shared_log_time = time.time() - start
        print("cuda stride shared log mul time: ", cuda_mul_shared_log_time)

        x = x.unsqueeze(dim=1)

        # warm-up
        expect = complex_mul(x, y)
        expect = expect.sum(dim=2)

        start = time.time()
        for _ in range(repeat):
            expect = complex_mul(x, y)
            expect = expect.sum(dim=2)
            torch.cuda.synchronize()
        pytorch_mul_time = time.time() - start
        print("broadcast mul time: ", pytorch_mul_time)

        print(f"pytorch is faster: {cuda_mul_time/pytorch_mul_time} X times")
        print(f"cuda is faster: {pytorch_mul_time / cuda_mul_time} X times")
        print(f"fft is faster than cuda multiply: {cuda_mul_time / pytorch_fft_time} X times")

        # print("out.size(): ", out.size())
        #
        # np.testing.assert_allclose(
        #     actual=out.cpu().numpy(), desired=expect.cpu().numpy(), rtol=1e-4,
        #     err_msg="actual out different from desired expected")



if __name__ == '__main__':
    unittest.main()