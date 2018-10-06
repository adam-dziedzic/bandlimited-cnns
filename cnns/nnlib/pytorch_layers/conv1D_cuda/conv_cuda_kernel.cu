#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// The size of block of threads.
const int blocksize = 1024;

//__device__ double atomicAdd(double *address, double val) {
//    unsigned long long int *address_as_ull =
//            (unsigned long long int *) address;
//    unsigned long long int old = *address_as_ull, assumed;
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//                        __double_as_longlong(val +
//                                             __longlong_as_double(assumed)));
//    } while (assumed != old);
//    return __longlong_as_double(old);
//}

template<typename scalar_t>
__global__ void plus_reduce_cuda_kernel(
        const scalar_t *input,
        const int64_t input_size,
        float *total_sum_ptr) {

    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Each block loads its elements into shared memory, padding with 0 if
    // input_size is not a multiple of blocksize.

    // Sharing the data for the block of threads.
    __shared__
    scalar_t x[blocksize];

    // For the last block we might have to add the zero elements at the end.
    x[tid] = (index < input_size) ? input[index] : 0;

    // Synchronize threads (this is a barrier) to update the cache - the same
    // coherent view of all the threads on the GPU.
    __syncthreads();

    // Every thread now holds 1 input value in x[].

    // Build summation tree over the elements (the blockDim.x is a power of 2).
    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        // Every thread holds sum of blocksize/s elements.
        if (tid < s) x[tid] += x[tid + s];
        __syncthreads();
    }

    // Thread 0 now holds the sum of all the input values to this block. Have it
    // add that sum to the running total. atomicAdd has no templating so we have
    // to define precisely the argument of at least one of the arguments.
    if (tid == 0) atomicAdd(total_sum_ptr, (float)x[0]);
}


float plus_reduce_cuda(at::Tensor input) {
    // at::Scalar total_sum = at::Scalar()
    // TensorOpions options = TensorOptions(input.device());
    // at::Tensor total_sum = at::zeros({1}, at::device(input.device()).dtype(input.scalar_type()));
    // void* total_sum_ptr = total_sum.data_ptr();
    float total_sum = 0.0;
    const int64_t input_size = input.size(0);

    const dim3 blocks((input_size + blocksize - 1) / blocksize, blocksize);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "plus_reduce_cuda", ([&] {
        plus_reduce_cuda_kernel<scalar_t><<< blocks, blocksize >>>
                                                       (input.data<scalar_t>(), input_size, &total_sum);
    }));
    // at::Tensor result = at::tensor({total_sum});
    return total_sum;
}

std::vector <at::Tensor> conv_cuda_forward(
        at::Tensor input,
        at::Tensor filter,
        at::Tensor bias,
        at::Tensor padding,
        at::Tensor index_back) {
    return {};
}

std::vector <at::Tensor> conv_cuda_backward(
        at::Tensor dout,
        at::Tensor xfft,
        at::Tensor yfft,
        at::Tensor W,
        at::Tensor WW,
        at::Tensor fft_size) {
    return {};
}