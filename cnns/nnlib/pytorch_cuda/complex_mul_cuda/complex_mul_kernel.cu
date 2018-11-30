#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

at::Tensor complex_mul_cuda(at::Tensor x, at::Tensor y) {
    const auto batch_size = x.size(0);

}