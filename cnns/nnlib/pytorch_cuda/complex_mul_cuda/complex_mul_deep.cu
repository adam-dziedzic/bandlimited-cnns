/**
CUDA based complex multiplication of tensors with summation over the input
channels fused into the multiplication.

Author: Adam Dziedzic, ady@uchicago.edu

To compile with nvcc and run the tests, just uncomment the ATen library with the
include line below, and the declaration of the function that is used by PyTorch,
namely: complex_mul_shared_log_cuda.

*/
#include <ATen/ATen.h>  // this to be uncommented to compile the file with nvcc

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cassert>

namespace {

/**
    Is n a power of 2?
*/
__device__ __forceinline__ bool is_power_2_device(int n) {
    return (n & (n - 1)) == 0;
}

/**
    Is n a power of 2?
*/
inline bool is_power_2_host(int n) {
    return (n & (n - 1)) == 0;
}

/**
Cache is with complex numbers. Sum the complex channels for a given pixel.

:param cache: an array of complex values to be summed for C consecutive complex
elements.
:param cache_index: the position of the thread in the cache.
:param C: number of channels.
*/
template <typename scalar_t>
__device__ __forceinline__ void sum_power_2_channels(
    scalar_t* cache, const int cache_index, const int C) {
    const int I = 2;  // complex number representation as 2 float numbers
    const int cache_index_I = cache_index*I;
    int c_I;
    for (int c = C / 2; c != 0; c /= 2) {
        if (cache_index < c) {
            c_I = c*I;
            cache[cache_index_I] += cache[cache_index_I + c_I];
            cache[cache_index_I + 1] += cache[cache_index_I + c_I + 1];
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__device__ __forceinline__ void sum_channels(
    scalar_t* cache, int cache_index, int C) {
    const int I = 2;  // complex number representation as 2 float numbers
    int c = C;  // C - number of all channels, c - still to be summed channels
    const int cache_index_I = cache_index*I;
    int c_I;
    while (c != 0) {
        bool is_write = false;  // should we sum the values up with this thread?
        if (c % 2 == 0 || c == 1) {
            c /= 2;
            if (cache_index < c) {
                is_write = true;
            }
        } else {
            c = (c+1)/2;
            if (cache_index < c - 1) {
                is_write = true;
            }
        }
        if (is_write) {
            c_I = c*I;
            cache[cache_index_I] += cache[cache_index_I + c_I];
            cache[cache_index_I + 1] += cache[cache_index_I + c_I + 1];
        }
        __syncthreads();
    }
}

/**
Complex multiplication of tensors using shared memory with barrier
synchronization and final summation across channels in the logarithmic manner.
This operation for a given channel in input and filter is just a dot product.

grid: L x F x N
L - size of a (flattened) plane in 2D or length of a 1D signal
N - number of input maps (batch size)
F - number of filters in the filter bank

input: N, L, C
filter: F, L, C
output: N, F, L

Compute the element wise complex multiplication for each thread in the block and
write the result to the shared memory. Then synchronize the threads and in the
log based fashion sum up the results for the current output pixel l through
its channels.
*/
template <typename scalar_t>
__global__ void complex_mul_cuda_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out) {
    // The size of the shared memory cache should be twice the number of threads
    // per block as we store the real and imaginary part of the result.
    extern __shared__ float cache[];   // cache for the result of the complex multiplication

    const int I = 2; // the last dimension for the complex number
        // The x index changes most rapidly.
    const int l = blockIdx.x; // the index in the output, position in the input map
    const int L = gridDim.x; // number of cells/pixels in the input map, Think of L as HxW.

    const int f = blockIdx.y; // current index of a filter from the filter bank
    const int F = gridDim.y; // number of filters in the filter bank

    const int n = blockIdx.z; // current index of an image/input map in the batch

    const int c = threadIdx.x; // channel number == thread number
    const int C = blockDim.x; // number of channels in input maps and filters

    // Index input map and filter.
    const int n_idx = n * L * C * I;  // input map (image) index
    const int f_idx = f * L * C * I;  // filter index
    const int l_idx =     l * C * I;  // plane index
    const int c_idx =         c * I;  // channel index

    // Index output map.
    const int no_idx = n * F * L * I; // output map (image) index
    const int fo_idx =     f * L * I; // output channel (filter) index
    const int lo_idx =         l * I; // output plane index

    // index in the input map
    const int N_idx = n_idx + l_idx + c_idx;
    // Index in the filter (the filter is of exactly the same size and
    // dimensions as the input map.
    int F_idx = f_idx + l_idx + c_idx;

    scalar_t out_re = 0;
    scalar_t out_im = 0;

    scalar_t x_re = x[N_idx];
    scalar_t x_im = x[N_idx + 1];
    scalar_t y_re = y[F_idx];
    scalar_t y_im = y[F_idx + 1];
    single_mul(x_re, x_im, y_re, y_im, &out_re, &out_im);

    cache[c_idx] = out_re;
    cache[c_idx + 1] = out_im;

    __syncthreads();  // Make the results visible to all threads.

    // Summed the pixels across channels.
    // It is of complexity O(logN). For each element in the output
    // map we add the computed pixels summed across channels.
    // This goes through all the channels present in the cache.

    if (is_power_2_device(C)) {
        sum_power_2_channels(/*cache=*/cache, /*cache_index=*/c, /*C=*/C);
    } else {
        sum_channels(/*cache=*/cache, /*cache_index=*/c, /*C=*/C);
    }

    // printf("thread_idx:%d, C:%d, N_idx:%d, last_N_idx:%d, cache_re:%d, cache_im:%d\n", threadIdx.x, C, N_idx, last_N_idx, cache[thread_cidx], cache[thread_cidx + 1]);
    // Write the output for the pixels summed (across channels).
    // printf("thread_idx:%d", threadIdx.x);
    if (threadIdx.x == 0) {
        // index in the output, we compute cells on a flat plane (no channels).
        int O_idx = no_idx + fo_idx + lo_idx;
        out[O_idx] = cache[0];
        out[O_idx + 1] = cache[1];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void single_mul(
    scalar_t x_re,
    scalar_t x_im,
    scalar_t y_re,
    scalar_t y_im,
    scalar_t* out_re,
    scalar_t* out_im) {

    scalar_t uavc = x_re * (y_re + y_im);
    *out_re += uavc - (x_re + x_im) * y_im;
    *out_im += (x_im - x_re) * y_re + uavc;
}

template <typename scalar_t>
void single_mul_simple(
    scalar_t x_re,
    scalar_t x_im,
    scalar_t y_re,
    scalar_t y_im,
    scalar_t* out_re,
    scalar_t* out_im) {

    *out_re += x_re * y_re - x_im * y_im;
    *out_im += x_re * y_im + x_im * y_re;
}

template <typename scalar_t>
__device__ __forceinline__ void single_add(
    scalar_t x_re,
    scalar_t x_im,
    scalar_t y_re,
    scalar_t y_im,
    scalar_t* out_re,
    scalar_t* out_im) {

    *out_re += x_re + y_re;
    *out_im += x_im + y_im;
}

template <typename scalar_t>
__global__ void test_sum_power_2_channels_device(scalar_t* cache, int C) {
    // printf("threadIdx.x: %d, C:%d\n", threadIdx.x, C);
    sum_power_2_channels(cache, threadIdx.x, C);
}

template <typename scalar_t>
__global__ void test_sum_channels_device(scalar_t* cache, int C) {
    // printf("threadIdx.x: %d, C:%d\n", threadIdx.x, C);
    sum_channels(cache, threadIdx.x, C);
}

void test_sum_channels_host(int C) {
    const int I = 2;
    int size_input = C*I;
    // Allocate unified memory - accessible from cpu or gpu
    // cudaMallocManaged(&x, size_input*sizeof(int));
    int *x = new int[size_input];
    int expect_re = 0;
    int expect_im = 0;
    int *d_x;
    cudaMalloc(&d_x, size_input*sizeof(int));

    for (int i = 0; i < size_input; ++i) {
        x[i] = i;
    }

    // printf("Expected numbers for the output map (after summing up channels):\n");
    // Generate the expected output y - only for each channel.
    int c_idx = 0;
    for (int c = 0; c < C; ++c) {
        c_idx = c*I;
        expect_re += x[c_idx];
        expect_im += x[c_idx + 1];
    }

    cudaMemcpy(d_x, x, size_input*sizeof(int), cudaMemcpyHostToDevice);

    const dim3 blocks(1);
    const int cuda_block_threads = C;

    if (is_power_2_host(C)) {
        test_sum_power_2_channels_device<int><<<blocks, cuda_block_threads>>>(d_x, C);
    } else {
        test_sum_channels_device<int><<<blocks, cuda_block_threads>>>(d_x, C);
    }

    // Wait for GPU to finish before accessing on host.
    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, size_input*sizeof(int), cudaMemcpyDeviceToHost);

    // printf("expect re: %d, expect im: %d\n", expect_re, expect_im);
    // printf("x re: %d, x im: %d\n", x[0], x[1]);

    assert (expect_re == x[0]);
    assert (expect_im == x[1]);

    cudaFree(d_x);
    cudaDeviceSynchronize();
    delete [] x;
}

void test_complex_multiply_host(int H, int W, int C) {
    const int I = 2;

    int size_input = H*W*C*I;
    int size_output = H*W*I;

    long long *x = new long long[size_input];
    long long *y = new long long[size_input];
    long long *out = new long long[size_output];
    long long *expect = new long long[size_output];

    std::size_t type_size = sizeof(long long);
    int raw_size_input = size_input * type_size;
    int raw_size_output = size_output * type_size;

    // Allocate unified memory - accessible from cpu or gpu
    // cudaMallocManaged(&x, size_input*sizeof(int));
    long long *d_x, *d_y, *d_out;  // memory on the device
    cudaMalloc(&d_x, raw_size_input);
    cudaMalloc(&d_y, raw_size_input);
    cudaMalloc(&d_out, raw_size_output);

    // set values for the input
//    for (int i = 0; i < size_input; ++i) {
//        x[i] = i;
//        y[i] = i;
//    }

    // Simple numbers 1 for the tests.
//    for (int i=0; i < size_input; i+=2) {
//        x[i] = 1;
//        x[i+1] = 0;
//        y[i] = 1;
//        y[i+1] = 0;
//    }

    // Random numbers for the tests.
    /* initialize random seed: */
    srand (time(NULL));
    /* generate random numbers between 0 and 10: */
    int range = 11;
    for (int i=0; i < size_input; i+=2) {
        x[i] = rand() % range;
        x[i+1] = rand() % range;
        y[i] = rand() % range;
        y[i+1] = rand() % range;
    }

    // zero out output
    for (int i = 0; i < size_output; ++i) {
        out[i] = 0;
        expect[i] = 0;
    }

//    printf("Initial numbers:\n");
//    for (int h=0; h<H; ++h) {
//        int h_idx = h*W*C*I;
//        for (int w=0; w<W; ++w) {
//            int w_idx = w*C*I;
//            for (int c=0; c<C; ++c) {
//               int c_idx = c*I;
//               printf("(h:%d, w:%d, c:%d): %lld + %lld \n", h, w, c,
//                   x[h_idx + w_idx + c_idx], x[h_idx + w_idx + c_idx + 1]);
//            }
//        }
//    }

    cudaMemcpy(d_x, x, raw_size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, raw_size_input, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_out, out, size_output*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemset(d_out, 0, raw_size_output);

    const dim3 blocks(/*L=*/H*W, 1, 1);
    const int block_threads = min(1024, C);
    // const int block_threads = 9;
    // printf("block threads: %d\n", block_threads);

    complex_mul_cuda_kernel<long long><<<blocks, block_threads,
        block_threads*2*type_size>>>(d_x, d_y, d_out);

    // Wait for GPU to finish before accessing on host.
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, raw_size_output, cudaMemcpyDeviceToHost);

    // printf("Expected numbers for the output map (after summing up channels):\n");
    // Generate the expected output y - only for each channel.
    for (int h=0; h<H; ++h) {
        int h_idx = h*W*C*I;
        int out_h_idx = h*W*I;
        for (int w=0; w<W; ++w) {
            int w_idx = w*C*I;   // channels are on the last but one dimension
            int out_w_idx = w*I;
            for (int c = 0; c < C; ++c) {
                int c_idx = c*I;
                int x_re = x[h_idx + w_idx + c_idx];
                int x_im = x[h_idx + w_idx + c_idx + 1];
                int y_re = y[h_idx + w_idx + c_idx];
                int y_im = y[h_idx + w_idx + c_idx + 1];
                int out_re = 0;
                int out_im = 0;
                single_mul_simple(x_re, x_im, y_re, y_im, &out_re, &out_im);

//                printf("x: (h: %d, w:%d, c:%d): %d + %dj\n", h, w, c,
//                   x[h_idx + w_idx + c_idx], x[h_idx + w_idx + c_idx + 1]);
//                printf("y: (h: %d, w:%d, c:%d): %d + %dj\n", h, w, c,
//                   y[h_idx + w_idx + c_idx], y[h_idx + w_idx + c_idx + 1]);
//                printf("out: (h: %d, w:%d, c:%d): %d + %dj\n", h, w, c,
//                   out_re, out_im);

                expect[out_h_idx + out_w_idx] += out_re;
                expect[out_h_idx + out_w_idx + 1] += out_im;
            }
            // Uncomment the printf below to debug the code.
            // printf("expected: (h:%d, w:%d): %lld + %lld\n", h, w, expect[out_h_idx + out_w_idx], expect[out_h_idx + out_w_idx + 1]);
            // printf("obtained: (h:%d, w:%d): %lld + %lld\n", h, w, out[out_h_idx + out_w_idx], out[out_h_idx + out_w_idx + 1]);
            assert (expect[out_h_idx + out_w_idx] == out[out_h_idx + out_w_idx]);
            assert (expect[out_h_idx + out_w_idx + 1] == out[out_h_idx + out_w_idx + 1]);
        }
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);

    cudaDeviceSynchronize();

    delete [] x;
    delete [] y;
    delete [] out;
    delete [] expect;
}

/**
Uncomment the pytorch related stuff.

Compile:
ady@skr-compute1:/tmp/pycharm_project_154/cnns/nnlib/pytorch_cuda/complex_mul_cuda$ nvcc complex_mul_kernel_stride_no_permute.cu -o complex_mul_profile.out
ady@skr-compute1:/tmp/pycharm_project_154/cnns/nnlib/pytorch_cuda/complex_mul_cuda$ nvprof ./complex_mul_profile.out

nvidia

/usr/local/cuda/bin/nvcc -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/TH -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/local/ady/anaconda3/include/python3.6m -c complex_mul_kernel.cu -o complex_mul_kernel_stride_no_permute.out -std=c++11
nvcc -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/TH -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/local/ady/anaconda3/include/python3.6m complex_mul_kernel_stride_no_permute.cu -o complex_mul_kernel_stride_no_permute.out -std=c++11
Segmentation fault
*/

void test_sum_channels_suit() {
    for (int c=0; c<1025; ++c) {
        test_sum_channels_host(/*C=*/c);
    }
    printf("finished test sum channels\n");
}

void test_multiply_suit() {
    test_complex_multiply_host(/*H=*/3, /*W=*/2, /*C=*/1);
    test_complex_multiply_host(/*H=*/3, /*W=*/2, /*C=*/3);
    test_complex_multiply_host(/*H=*/16, /*W=*/8, /*C=*/4);
    test_complex_multiply_host(/*H=*/1, /*W=*/1, /*C=*/64);
    test_complex_multiply_host(/*H=*/2, /*W=*/1, /*C=*/512);
    test_complex_multiply_host(/*H=*/2, /*W=*/2, /*C=*/512);
    test_complex_multiply_host(/*H=*/32, /*W=*/32, /*C=*/3);

    // ResNet sizes: 100% energy preserved.
    test_complex_multiply_host(/*H=*/128, /*W=*/65, /*C=*/3);
    test_complex_multiply_host(/*H=*/64, /*W=*/33, /*C=*/64);
    test_complex_multiply_host(/*H=*/32, /*W=*/17, /*C=*/128);
    test_complex_multiply_host(/*H=*/16, /*W=*/9, /*C=*/256);
    test_complex_multiply_host(/*H=*/8, /*W=*/5, /*C=*/512);

    // ResNet sizes: 95% energy preserved.
    test_complex_multiply_host(/*H=*/85, /*W=*/43, /*C=*/3);
    test_complex_multiply_host(/*H=*/43, /*W=*/22, /*C=*/64);
    test_complex_multiply_host(/*H=*/21, /*W=*/11, /*C=*/128);
    test_complex_multiply_host(/*H=*/15, /*W=*/8, /*C=*/256);
    test_complex_multiply_host(/*H=*/8, /*W=*/5, /*C=*/512);

    // ResNet-18 sizes: 99% energy preserved.
    // xfft size:  torch.Size([128, 3, 119, 60, 2])
    // xfft size:  torch.Size([128, 64, 61, 31, 2])
    // xfft size:  torch.Size([128, 64, 59, 30, 2])
    // xfft size:  torch.Size([128, 64, 59, 30, 2])
    // xfft size:  torch.Size([128, 64, 57, 29, 2])
    // xfft size:  torch.Size([128, 64, 57, 29, 2])
    // xfft size:  torch.Size([128, 128, 32, 17, 2])
    // xfft size:  torch.Size([128, 128, 31, 16, 2])
    // xfft size:  torch.Size([128, 128, 31, 16, 2])
    // xfft size:  torch.Size([128, 128, 31, 16, 2])
    // xfft size:  torch.Size([128, 256, 16, 9, 2])
    // xfft size:  torch.Size([128, 256, 16, 9, 2])
    // xfft size:  torch.Size([128, 256, 16, 9, 2])
    // xfft size:  torch.Size([128, 256, 16, 9, 2])
    // xfft size:  torch.Size([128, 512, 8, 5, 2])
    // xfft size:  torch.Size([128, 512, 8, 5, 2])
    // xfft size:  torch.Size([128, 512, 8, 5, 2])
    test_complex_multiply_host(/*H=*/119, /*W=*/60, /*C=*/3);
    test_complex_multiply_host(/*H=*/61, /*W=*/31, /*C=*/64);
    test_complex_multiply_host(/*H=*/59, /*W=*/30, /*C=*/64);
    test_complex_multiply_host(/*H=*/57, /*W=*/29, /*C=*/64);
    test_complex_multiply_host(/*H=*/32, /*W=*/17, /*C=*/128);
    test_complex_multiply_host(/*H=*/31, /*W=*/16, /*C=*/128);
    test_complex_multiply_host(/*H=*/21, /*W=*/11, /*C=*/128);
    test_complex_multiply_host(/*H=*/16, /*W=*/9, /*C=*/256);
    test_complex_multiply_host(/*H=*/8, /*W=*/5, /*C=*/512);

    // ResNet-18 sizes: 90// % energy preserved.
    // xfft size:  torch.Size([128, 3, 59, 30, 2])
    // xfft size:  torch.Size([128, 64, 29, 15, 2])
    // xfft size:  torch.Size([128, 64, 21, 11, 2])
    // xfft size:  torch.Size([128, 64, 21, 11, 2])
    // xfft size:  torch.Size([128, 64, 15, 8, 2])
    // xfft size:  torch.Size([128, 64, 13, 7, 2])
    // xfft size:  torch.Size([128, 128, 11, 6, 2])
    // xfft size:  torch.Size([128, 128, 11, 6, 2])
    // xfft size:  torch.Size([128, 128, 9, 5, 2])
    // xfft size:  torch.Size([128, 128, 9, 5, 2])
    // xfft size:  torch.Size([128, 256, 7, 4, 2])
    // xfft size:  torch.Size([128, 256, 9, 5, 2])
    // xfft size:  torch.Size([128, 256, 9, 5, 2])
    // xfft size:  torch.Size([128, 256, 9, 5, 2])
    // xfft size:  torch.Size([128, 512, 8, 5, 2])
    // xfft size:  torch.Size([128, 512, 7, 4, 2])
    // xfft size:  torch.Size([128, 512, 8, 5, 2])
    test_complex_multiply_host(/*H=*/59, /*W=*/30, /*C=*/3);
    test_complex_multiply_host(/*H=*/29, /*W=*/15, /*C=*/64);
    test_complex_multiply_host(/*H=*/21, /*W=*/11, /*C=*/64);
    test_complex_multiply_host(/*H=*/15, /*W=*/8, /*C=*/64);
    test_complex_multiply_host(/*H=*/13, /*W=*/7, /*C=*/64);
    test_complex_multiply_host(/*H=*/11, /*W=*/8, /*C=*/128);
    test_complex_multiply_host(/*H=*/9, /*W=*/5, /*C=*/128);
    test_complex_multiply_host(/*H=*/7, /*W=*/4, /*C=*/256);
    test_complex_multiply_host(/*H=*/9, /*W=*/5, /*C=*/256);
    test_complex_multiply_host(/*H=*/7, /*W=*/4, /*C=*/512);
    test_complex_multiply_host(/*H=*/8, /*W=*/5, /*C=*/512);
    printf("Finished test multiplication.\n");
}

} // namespace

/**
:param: The out tensor has to be zero-initialized.

This to be uncommented to compile the file with nvcc.
*/
void complex_mul_deep_cuda(
    at::Tensor x,
    at::Tensor y,
    at::Tensor out) {

    const auto N = x.size(0);  // batch_size
    const auto F = y.size(0);  // filter_bank_size

    if (x.sizes().size() == 4) { // 2D data
        const auto H = x.size(1);  // height of the matrix
        const auto W = x.size(2);  // width of the matrix
        const auto C = x.size(3);  // number of channels
        const auto L = H*W;
    } else {  // 1D data
        const auto L = x.size(1);  // height of the matrix
        const auto C = x.size(2);  // number of channels
    }

    const dim3 blocks(/*L=*/L, F, N);
    const int block_threads = min(1024L, C);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "complex_mul_cuda",
    ([&] {
        complex_mul_cuda_kernel<scalar_t><<<blocks, block_threads,
        block_threads*2*sizeof(scalar_t)>>>(
            x.data<scalar_t>(), y.data<scalar_t>(), out.data<scalar_t>());
    }));
}

// Uncomment main function for tests.
//int main(void)
//{
//    test_sum_channels_suit();
//    test_multiply_suit();
//    printf("All tests finished successfully.\n");
//    return 0;
//}