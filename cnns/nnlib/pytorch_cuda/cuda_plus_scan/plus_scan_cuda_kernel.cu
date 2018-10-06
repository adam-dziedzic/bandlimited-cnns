#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{

template <typename scalar_t>
__global__ void plus_scan_cuda_forward_kernel(

}

} // namespace

at::Tensor plus_scan_cuda(
    at::Tensor input) {

}
