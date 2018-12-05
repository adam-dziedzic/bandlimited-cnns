// #include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <cassert>

namespace {

/**
Complex multiplication of tensors using shared memory with barrier
synchronization and final summation across channels in the logarithmic manner.

Compute the element wise complex multiplication for each thread in the block and
write the result to the shared memory. Then synchronize the threads and in the
log based fashion sum up the results for each output pixel through its channels,
if they are present in the cache. The stride is the number of threads per block
times the I (the two float representation of complex numbers).
*/
template <typename scalar_t>
__global__ void complex_mul_cuda_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int N, const int F, const int H, const int W, const int C) {
    // The size of the shared memory cache should be twice the number of threads
    // per block as we store the real and imaginary part of the result.
    extern __shared__ float cache[];   // cache for the result of the complex multiplication

    const int I = 2; // the last dimension for the complex number
    const int plane_size = H * W;
    const int channel_size = C * I;
    const int image_size = plane_size * channel_size;  // size of the image from the batch
    // number of complex values in the input that we iterate through
    const int nr_values = H * W * C;

    const int n = blockIdx.x; // current index of an image/input map in the batch
    const int f = blockIdx.y; // current index of a filter from the filter bank
    const int block_size = blockDim.x;

    // After running all the threads in the block, we increment the thread
    // number by the block size.
    int thread_nr = threadIdx.x;

    // stride for the H*W map is equal to the number of threads declared in a block
    const int stride = block_size * I; // we need H*W threads per plane, each deals with I numbers

    const int n_idx = n * image_size;  // start index in the batch for this input map
    const int f_idx = f * image_size;  // start index in the bank for this filter

    // find index for the output
    const int output_fchannel_size = plane_size * I;
    const int no_idx = n * (F * output_fchannel_size); // output index for the batch data point
    const int fo_idx = f * output_fchannel_size;       // output index for the filter/channel

    // We go through each complex number one by one.
    // We linearize it and start from 0, move by #threads*I steps in outer loop.
    const int start_idx = threadIdx.x*I;

    // index in the input map
    int N_idx = n_idx + start_idx; // index across the first channel plane (in the input map n).
    const int last_N_idx = n_idx + image_size;  // last index for the starting position to compute the sum through each channel for this pixel

    // To prevent us from a deadlock, we have to always execute __syncthreads();
    // for all the threads in the block. Each thread has to do the same number of
    // iterations for any loop. To ensure that, we keep all threads running,
    // even though, some of them are really idle. We keep the loop running to
    // the multiple of the block size that is greater than the number of values
    // in the input map in total: C*H*W - this is a number of complex cells in the
    // input map.
    const int num_blocks = (nr_values + block_size - 1) / block_size;
    const int last_block_idx = n_idx + num_blocks * block_size * I;

    // Index in the filter (the filter is of exactly the same size and
    // dimensions as the input map.
    int F_idx = f_idx + start_idx;

    // index in the output, we compute cells on a flat plane (no channels).
    int base_O_idx = no_idx + fo_idx;

    // Cache (c) index;
    int thread_cidx = thread_nr * I;

    printf("N_idx:%d, last_block_idx:%d, last_N_idx:%d\n", N_idx, last_block_idx, last_N_idx);

    while (N_idx < last_block_idx)  {

        // Zero out caches.
        cache[thread_cidx] = 0;
        cache[thread_cidx + 1] = 0;

        if (N_idx < last_N_idx - 1) {
            scalar_t out_re = 0;
            scalar_t out_im = 0;

            scalar_t x_re = x[N_idx];
            scalar_t x_im = x[N_idx + 1];
            scalar_t y_re = y[F_idx];
            scalar_t y_im = y[F_idx + 1];
            single_mul(x_re, x_im, y_re, y_im, &out_re, &out_im);

            cache[thread_cidx] = out_re;
            cache[thread_cidx + 1] = out_im;
        }

        __syncthreads();  // Make the results visible to all threads.

        // Summed the pixels across channels.
        // It is of complexity O(logN). For each element in the output
        // map we add the computed pixels summed across channels.
        // This goes through all the channels present in the cache.

        sum_channels(cache, threadIdx.x, C);

        __syncthreads();
        // Write the output for the pixels summed (across channels).
        if (threadIdx.x % C == 0) {
            // Running index through the current XY plane in the output.
            // Assume 8 threads. Plane size HxW = 3x2 = 6 and 3 channels.
            // There are 6*3=18 complex multiplications for the output f-th plane.
            // The 0th, 1st, and 2nd threads should return 0 index in the output.
            // 3rd,4th,5th threads should return 1st index in the output.
            // The 20th index should return
            int run_O_idx = (thread_nr % (plane_size * C)) / C;

            const int O_idx = base_O_idx + run_O_idx*I;
            out[O_idx] += cache[thread_cidx];
            out[O_idx + 1] = cache[thread_cidx + 1];
        }

        N_idx += stride;
        F_idx += stride;
        thread_nr += block_size;

        // Make sure that all cache cells are zeroed out before moving on.
        // We need this as in the second part we access cache cells that do not
        // belong only to this thread.
        __syncthreads();
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

/**
Cache is with complex numbers. Sum the complex channels for a given pixel.

:param cache: an array of complex values to be summed for C consecutive complex
elements.
:param cache_index: the position of the thread in the cache.
:param C: number of channels.
*/
template <typename scalar_t>
__device__ __forceinline__
void sum_channels(scalar_t* cache, int cache_index, int C) {
    const int I = 2;  // complex number representation as 2 float numbers
    int c = C;  // C - number of all channels, c - still to be summed channels
    while (c != 0) {
        // printf("cache_index:%d, c:%d\n", cache_index, c);
        bool is_write = false;  // should we sum the values up with this thread?
        if (c % 2 == 0 || c == 1) {
            c /= 2;
            if (cache_index % C < c) {
                is_write = true;
            }
        } else {
            c = (c+1)/2;
            if (cache_index % C < c - 1) {
                is_write = true;
            }
        }
        if (is_write) {
            const int cache_index_I = cache_index*I;
            const int c_I = c*I;
            cache[cache_index_I] += cache[cache_index_I + c_I];
            cache[cache_index_I + 1] += cache[cache_index_I + c_I + 1];
        }
        __syncthreads();
        // printf("%d: %d, %d\n", cache_index, cache[cache_index*I], cache[cache_index*I+1]);
    }
}

template <typename scalar_t>
__global__ void test_sum_channels_device(scalar_t* cache, int C) {
    // printf("threadIdx.x: %d, C:%d\n", threadIdx.x, C);
    sum_channels(cache, threadIdx.x, C);
}

void test_sum_channels_host() {
    int *x, *y;
    const int I = 2;
    const int C = 2;
    const int W = 3;
    int size_input = W*C*I;

    // Allocate unified memory - accessible from cpu or gpu
    // cudaMallocManaged(&x, size_input*sizeof(int));
    x = new int[size_input];
    y = new int[size_input];
    int *d_x;
    cudaMalloc(&d_x, size_input*sizeof(int));

    for (int i = 0; i < size_input; ++i) {
        x[i] = i;
        y[i] = i;
    }

    printf("Initial numbers:\n");
    for (int w=0; w<W; ++w) {
        int w_idx = w*C*I;
        for (int c=0; c<C; ++c) {
           int c_idx = c*I;
           printf("(w:%d, c:%d): %d + %dj\n", w, c, x[w_idx + c_idx], x[w_idx+c_idx+1]);
        }
    }

    cudaMemcpy(d_x, x, size_input*sizeof(int), cudaMemcpyHostToDevice);

    const dim3 blocks(1);
    const int cuda_block_threads = 6;

    test_sum_channels_device<int><<<blocks, cuda_block_threads>>>(d_x, C);

    // Wait for GPU to finish before accessing on host.
    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, size_input*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Expected numbers for the output map (after summing up channels):\n");
    // Generate the expected output y - only for each channel.
    for (int w=0; w<W; ++w) {
        int w_idx = w*C*I;   // channels are on the last but one dimension
        printf("(w:%d): %d + %dj\n", w, y[w_idx], y[w_idx + 1]);
        for (int c = 1; c < C; ++c) {
            int c_idx = c*I;
            printf("(w:%d, c:%d): %d + %dj\n", w, c, y[w_idx + c_idx], y[w_idx + c_idx + 1]);
            y[w_idx] += y[w_idx + c_idx];
            y[w_idx + 1] += y[w_idx + c_idx + 1];
        }
        printf("expected: (w:%d): %d + %dj\n", w, y[w_idx], y[w_idx + 1]);
        printf("obtained: (w:%d): %d + %dj\n", w, x[w_idx], x[w_idx + 1]);
        assert (y[w_idx] == x[w_idx]);
        assert (y[w_idx + 1] == x[w_idx + 1]);
    }

    printf("Obtained numbers:\n");
    for (int w=0; w<W; ++w) {
        int w_idx = w*C*I;
        for (int c=0; c<C; ++c) {
           int c_idx = c*I;
           printf("(w:%d, c:%d): %d + %dj\n", w, c, x[w_idx + c_idx], x[w_idx+c_idx+1]);
        }
    }

//    for (int i=0; i < size_input; ++i) {
//        printf("%d: %d\n", i, x[i]);
//        // assert (expect[i] == x[i++]);
//    }

    cudaFree(d_x);
    cudaDeviceSynchronize();
    delete [] x;
    delete [] y;

    printf("finished test sum channels\n");
}

void test_sum_channels_host_big(int C, int W) {
    int *x, *y;
    const int I = 2;

    int size_input = W*C*I;

    // Allocate unified memory - accessible from cpu or gpu
    // cudaMallocManaged(&x, size_input*sizeof(int));
    x = new int[size_input];
    y = new int[size_input];
    int *d_x;
    cudaMalloc(&d_x, size_input*sizeof(int));

    for (int i = 0; i < size_input; ++i) {
        x[i] = i;
        y[i] = i;
    }

    printf("Initial numbers:\n");
    for (int w=0; w<W; ++w) {
        int w_idx = w*C*I;
        for (int c=0; c<C; ++c) {
           int c_idx = c*I;
           printf("(w:%d, c:%d): %d + %dj\n", w, c, x[w_idx + c_idx], x[w_idx+c_idx+1]);
        }
    }

    cudaMemcpy(d_x, x, size_input*sizeof(int), cudaMemcpyHostToDevice);

    const dim3 blocks(1);
    const int cuda_block_threads = W*C;

    test_sum_channels_device<int><<<blocks, cuda_block_threads>>>(d_x, C);

    // Wait for GPU to finish before accessing on host.
    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, size_input*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Expected numbers for the output map (after summing up channels):\n");
    // Generate the expected output y - only for each channel.
    for (int w=0; w<W; ++w) {
        int w_idx = w*C*I;   // channels are on the last but one dimension
        for (int c = 1; c < C; ++c) {
            int c_idx = c*I;
            y[w_idx] += y[w_idx + c_idx];
            y[w_idx + 1] += y[w_idx + c_idx + 1];
        }
        printf("expected: (w:%d): %d + %dj\n", w, y[w_idx], y[w_idx + 1]);
        printf("obtained: (w:%d): %d + %dj\n", w, x[w_idx], x[w_idx + 1]);
        assert (y[w_idx] == x[w_idx]);
        assert (y[w_idx + 1] == x[w_idx + 1]);
    }

    printf("Obtained numbers:\n");
    for (int w=0; w<W; ++w) {
        int w_idx = w*C*I;
        for (int c=0; c<C; ++c) {
           int c_idx = c*I;
           printf("(w:%d, c:%d): %d + %dj\n", w, c, x[w_idx + c_idx], x[w_idx+c_idx+1]);
        }
    }

    cudaFree(d_x);
    cudaDeviceSynchronize();
    delete [] x;
    delete [] y;

    printf("finished test sum channels\n");
}

} // namespace

//void complex_mul_shared_log_cuda(
//    at::Tensor x,
//    at::Tensor y,
//    at::Tensor out) {
//
//    const auto N = x.size(0);  // batch_size
//    const auto F = y.size(0);  // filter_bank_size
//    const auto H = x.size(1);  // height of the matrix
//    const auto W = x.size(2);  // width of the matrix
//    const auto C = x.size(3);  // number of channels
//
//    const auto x_blocks = N;
//    const auto y_blocks = F;
//    const dim3 blocks(x_blocks, y_blocks);
//
//    const int threads = int(1024/C) * C;
//
//    AT_DISPATCH_FLOATING_TYPES(x.type(), "complex_mul_cuda",
//    ([&] {
//        complex_mul_cuda_kernel<scalar_t><<<blocks, threads>>>(
//        x.data<scalar_t>(), y.data<scalar_t>(), out.data<scalar_t>(),
//        N, F, C, H, W);
//    }));
//}

//template <typename scalar_t>
//void complex_mul_stride_no_permute_cuda_pure(
//    at::Tensor x,
//    at::Tensor y,
//    at::Tensor out,
//    int threads = 1024) {
//
//    const auto N = x.size(0);  // batch_size
//    const auto F = y.size(0);  // filter_bank_size
//    const auto C = x.size(1);  // number of channels
//    const auto H = x.size(2);  // height of the matrix
//    const auto W = x.size(3);  // width of the matrix
//
//    const auto x_blocks = N;
//    const auto y_blocks = F;
//    const dim3 blocks(x_blocks, y_blocks);
//
//    // Run kernel on the GPU
//    complex_mul_cuda_kernel<scalar_t><<<blocks, 1024>>>(
//        x.data<scalar_t>(), y.data<scalar_t>(), out.data<scalar_t>(),
//        N, F, C, H, W);
//}

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

void test_multiply() {
    int N = 1;
    int F = 1;
    int H = 3;
    int W = 2;
    int C = 3;
    int I = 2;
    int size_input = N * H * W * C * I;
    int size_filter = F * H * W * C * I;
    int size_output = N * F * H * W * I;
    int cuda_block_threads = int(16/3) * 3;

    // auto dims = {128, 32, 16, 8, 2};
    //    at::Tensor x = at::randn({128, 32, 16, 8, 2});
    //    at::Tensor y = at::randn({128, 32, 16, 8, 2});
    //    at::Tensor out = at::zeros({128, 32, 16, 8, 2});
    float *x, *y, * out;

    // Allocate unified memory - accessible from cpu or gpu
    cudaMallocManaged(&x, size_input*sizeof(float));
    cudaMallocManaged(&y, size_filter*sizeof(float));
    cudaMallocManaged(&out, size_output*sizeof(float));

    for (int j=0; j<H; ++j) {
        for (int i=0; i<W; ++i) {
            for (int c=0; c<C; ++c) {
                const int index = (j*W*C+i*C+c)*2;
                x[index] = index;
                x[index + 1] = index + 1;
                y[index] = 4;
                y[index + 1] = 2;
            }
        }
    }

    for (int i=0; i<H*W*2; i+=2) {
        printf("%p %d: %f, %f, %f, %f\n", x, i, x[i], x[i+1], y[i], y[i+1]);
    }

    const dim3 blocks(N, F);

    complex_mul_cuda_kernel<float><<<blocks, cuda_block_threads,
        cuda_block_threads*2>>>(x, y, out, N, F, H, W, C);

    for (int i=0; i<H*W*C; i+=2) {
        printf("%d: %f, %f\n", i, out[i], out[i+1]);
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(out);

    printf("finished computation\n");
}

void test_sum_channels_suit() {
    test_sum_channels_host();
    test_sum_channels_host_big(/*C=*/1, /*W=*/7);
    test_sum_channels_host_big(/*C=*/1, /*W=*/1024);
    test_sum_channels_host_big(/*C=*/3, /*W=*/300);
    test_sum_channels_host_big(/*C=*/1, /*W=*/19);
    test_sum_channels_host_big(/*C=*/4, /*W=*/4);
    test_sum_channels_host_big(/*C=*/16, /*W=*/6);
    test_sum_channels_host_big(/*C=*/32, /*W=*/32);
    test_sum_channels_host_big(/*C=*/3, /*W=*/300);
}

int main(void)
{
    test_sum_channels_suit();
    // test_multiply();

    return 0;
}