import torch
import unittest
from cnns.nnlib.pytorch_layers.pytorch_utils import flip
from cnns.nnlib.pytorch_layers.pytorch_utils import preserve_energy2D
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul
import numpy as np
from torch import tensor
import socket
import time

if socket.gethostname() == "skr-compute1":
    from complex_mul_cpp import complex_mul as complex_mul_cpp
    from complex_mul_cuda import complex_mul as complex_mul_cuda
    from complex_mul_cuda import complex_mul_stride as complex_mul_stride_cuda
    from complex_mul_cuda import \
        complex_mul_stride_no_permute as complex_mul_stride_no_permute_cuda


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

if __name__ == '__main__':
    unittest.main()