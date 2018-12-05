// #include <ATen/ATen.h>

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
    int thread_cidx = threadIdx.x * I;

    // printf("N_idx:%d, last_block_idx:%d, last_N_idx:%d\n", N_idx, last_block_idx, last_N_idx);

    while (N_idx < last_block_idx)  {
        // Check if N_idx is still within the bound for the plane of size: H*W*C
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
            // printf("thread_idx:%d, N_idx:%d, last_N_idx:%d, out_re:%d, out_im:%d\n", threadIdx.x, N_idx, last_N_idx, cache[thread_cidx], cache[thread_cidx + 1]);
        }

        __syncthreads();  // Make the results visible to all threads.

        // Summed the pixels across channels.
        // It is of complexity O(logN). For each element in the output
        // map we add the computed pixels summed across channels.
        // This goes through all the channels present in the cache.

        sum_channels(/*cache=*/cache, /*cache_index=*/threadIdx.x, /*C=*/C);

        // printf("thread_idx:%d, C:%d, N_idx:%d, last_N_idx:%d, cache_re:%d, cache_im:%d\n", threadIdx.x, C, N_idx, last_N_idx, cache[thread_cidx], cache[thread_cidx + 1]);
        // Write the output for the pixels summed (across channels).
        // printf("thread_idx:%d", threadIdx.x);
        if (threadIdx.x % C == 0 && N_idx < last_N_idx - 1) {
            // Running index through the current XY plane in the output.
            // Assume 8 threads. Plane size HxW = 3x2 = 6 and 3 channels.
            // There are 6*3=18 complex multiplications for the output f-th plane.
            // The 0th, 1st, and 2nd threads should return 0 index in the output.
            // 3rd,4th,5th threads should return 1st index in the output.
            // The 20th index should return
            int run_O_idx = thread_nr / C;

            const int O_idx = base_O_idx + run_O_idx*I;

            // printf("N_idx:%d, last_N_idx:%d, cache_re:%d, cache_im:%d\n", N_idx, last_N_idx, cache[thread_cidx], cache[thread_cidx + 1]);
            out[O_idx] += cache[thread_cidx];
            out[O_idx + 1] += cache[thread_cidx + 1];
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
//        printf("cache_index:%d, c:%d\n", cache_index, c);
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
    /* generate numbers between 0 and 10: */
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

    printf("Initial numbers:\n");
    for (int h=0; h<H; ++h) {
        int h_idx = h*W*C*I;
        for (int w=0; w<W; ++w) {
            int w_idx = w*C*I;
            for (int c=0; c<C; ++c) {
               int c_idx = c*I;
               printf("(h:%d, w:%d, c:%d): %lld + %lld \n", h, w, c,
                   x[h_idx + w_idx + c_idx], x[h_idx + w_idx + c_idx + 1]);
            }
        }
    }

    cudaMemcpy(d_x, x, raw_size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, raw_size_input, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_out, out, size_output*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, raw_size_output);

    const dim3 blocks(1);
    const int block_threads = min(int(1024/C) * C, H*W*C);
    // const int block_threads = 9;
    printf("block threads: %d\n", block_threads);

    complex_mul_cuda_kernel<long long><<<blocks, block_threads,
        block_threads*2*type_size>>>(
        d_x, d_y, d_out, /*N=*/1, /*F*=*/1, /*H=*/H, /*W=*/W, /*C=*/C);

    // Wait for GPU to finish before accessing on host.
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, raw_size_output, cudaMemcpyDeviceToHost);

    printf("Expected numbers for the output map (after summing up channels):\n");
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
            printf("expected: (h:%d, w:%d): %lld + %lld\n", h, w, expect[out_h_idx + out_w_idx], expect[out_h_idx + out_w_idx + 1]);
            printf("obtained: (h:%d, w:%d): %lld + %lld\n", h, w, out[out_h_idx + out_w_idx], out[out_h_idx + out_w_idx + 1]);
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

    printf("Finished test multiplication.\n");
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
//    const int threads = min(int(1024/C) * C, H*W*C);
//
//    cudaMemset(out, 0, N*F*H*W*sizeof(scalar_t));
//
//    AT_DISPATCH_FLOATING_TYPES(x.type(), "complex_mul_cuda",
//    ([&] {
//        complex_mul_cuda_kernel<scalar_t><<<blocks, threads, 2*threads*sizeof(scalar_t)>>>(
//        x.data<scalar_t>(), y.data<scalar_t>(), out.data<scalar_t>(),
//        N, F, C, H, W);
//    }));
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

    printf("Finished multiplication test.\n");
}

void test_sum_channels_suit() {
//    test_sum_channels_host();
//    test_sum_channels_host_big(/*C=*/64, /*W=*/1);
    test_sum_channels_host_big(/*C=*/3, /*W=*/341);
//    test_sum_channels_host_big(/*C=*/1, /*W=*/7);
//    test_sum_channels_host_big(/*C=*/1, /*W=*/1024);
//    test_sum_channels_host_big(/*C=*/3, /*W=*/300);
//    test_sum_channels_host_big(/*C=*/1, /*W=*/19);
//    test_sum_channels_host_big(/*C=*/4, /*W=*/4);
//    test_sum_channels_host_big(/*C=*/16, /*W=*/6);
//    test_sum_channels_host_big(/*C=*/32, /*W=*/32);
//    test_sum_channels_host_big(/*C=*/3, /*W=*/300);
}

void test_multiply_suit() {
//    test_complex_multiply_host(/*H=*/3, /*W=*/2, /*C=*/1);
//    test_complex_multiply_host(/*H=*/3, /*W=*/2, /*C=*/3);
//    test_complex_multiply_host(/*H=*/16, /*W=*/8, /*C=*/4);
//    test_complex_multiply_host(/*H=*/1, /*W=*/1, /*C=*/64);
//    test_complex_multiply_host(/*H=*/2, /*W=*/1, /*C=*/512);
//    test_complex_multiply_host(/*H=*/2, /*W=*/2, /*C=*/512);
    test_complex_multiply_host(/*H=*/32, /*W=*/32, /*C=*/3);
}

int main(void)
{
    // test_sum_channels_suit();
    test_multiply_suit();
    printf("All tests finished successfully.\n");
    return 0;
}