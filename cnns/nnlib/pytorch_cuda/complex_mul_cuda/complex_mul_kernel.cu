#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace {


template <typename scalar_t>
complex_mul_cuda_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y) {



}


} // namespace

void complex_mul_cuda(at::Tensor x, at::Tensor y, at::Tensor out) {

    const int threads = 1024;
    const auto batch_size = x.size(0);



}
