#ifndef COMPLEX_MUL_H
#define COMPLEX_MUL_H

#include <torch/torch.h>

// CUDA forward declarations

void complex_mul_cuda(at::Tensor x, at::Tensor y, at::Tensor out);
void complex_mul_stride_cuda(at::Tensor x, at::Tensor y, at::Tensor out);
void complex_mul_stride_no_permute_cuda(
    at::Tensor x, at::Tensor y, at::Tensor out, int threads = 1024);
void complex_mul_stride_no_permute_cuda_pure(
    at::Tensor x, at::Tensor y, at::Tensor out, int threads = 1024);

#endif  // COMPLEX_MUL_h