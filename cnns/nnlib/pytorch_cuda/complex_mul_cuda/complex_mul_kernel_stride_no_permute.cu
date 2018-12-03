#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace {

template <typename scalar_t>
__global__ void complex_mul_cuda_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int N, const int F, const int C, const int H, const int W) {

    const int I = 2; // the last dimension for the complex number
    const int plane_size = H * W;
    const int channel_size = plane_size * I;
    const int image_size = C * channel_size;  // size of the image from the batch

    const int n = blockIdx.x; // current index of an image in the batch
    const int f = blockIdx.y; // current index of a filter from the filter bank
    const int start = threadIdx.x; // current XY cell to be computed (element wise multiplication between corresponding cells in the image and filter, then summed up
    // number of threads per block
    const int raw_stride = blockDim.x;  // stride for the H*W map is equal to the number of threads declared in a block

    const int n_idx = n * image_size;  // start index in the batch for this input map
    const int f_idx = f * image_size;  // start index in the bank for this filter

    // find index for the output
    const int no_idx = n * (F * channel_size); // output index for the batch data point
    const int fo_idx = f * channel_size;       // output index for the filter/channel


    for (int raw_pixel = start; raw_pixel < plane_size; raw_pixel += raw_stride)  {
        /* If the plane is of size HxW = 32x32, and the raw_pixel is 45, then
        the h is 45 / 32 = 1, and the W (current column) is 45 - 1*32 = 13. */
        const int h = raw_pixel / plane_size;  // current row in the H,W,C,I plane
        const int w = raw_pixel - h * H;       // current col in the W,C,I linear space
        const int h_idx = h * W * I;   // start index for this row
        const int w_idx = w * I;           // start index for this column

        // index in the input map
        int N_idx = n_idx + h_idx + w_idx; // index for this C,I component in input

        // index in the filter
        int F_idx = f_idx + h_idx + w_idx; // index for this C,I component in filter

        // find the final index (last mile) for the output
        const int ho_idx = h * W * I;   // output index for row
        const int wo_idx = w * I;       // output index for col
        const int O_idx = no_idx + fo_idx + ho_idx + wo_idx;

        scalar_t out_re = 0.0;
        scalar_t out_im = 0.0;

        for (int c = 0; c < C; ++c) {
            scalar_t x_re = x[N_idx];
            scalar_t x_im = x[N_idx + 1];
            scalar_t y_re = y[F_idx];
            scalar_t y_im = y[F_idx + 1];
            N_idx += channel_size;
            F_idx += channel_size;
            single_mul(x_re, x_im, y_re, y_im, &out_re, &out_im);
        }
        out[O_idx] = out_re;
        out[O_idx + 1] = out_im;
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

} // namespace

void complex_mul_stride_no_permute_cuda(
    at::Tensor x,
    at::Tensor y,
    at::Tensor out,
    int threads = 1024) {

    const auto N = x.size(0);  // batch_size
    const auto F = y.size(0);  // filter_bank_size
    const auto C = x.size(1);  // number of channels
    const auto H = x.size(2);  // height of the matrix
    const auto W = x.size(3);  // width of the matrix

    const auto x_blocks = N;
    const auto y_blocks = F;
    const dim3 blocks(x_blocks, y_blocks);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "complex_mul_cuda",
    ([&] {
        complex_mul_cuda_kernel<scalar_t><<<blocks, threads>>>(
        x.data<scalar_t>(), y.data<scalar_t>(), out.data<scalar_t>(),
        N, F, C, H, W);
    }));
}

template <typename scalar_t>
void complex_mul_stride_no_permute_cuda_pure(
    at::Tensor x,
    at::Tensor y,
    at::Tensor out,
    int threads = 1024) {

    const auto N = x.size(0);  // batch_size
    const auto F = y.size(0);  // filter_bank_size
    const auto C = x.size(1);  // number of channels
    const auto H = x.size(2);  // height of the matrix
    const auto W = x.size(3);  // width of the matrix

    const auto x_blocks = N;
    const auto y_blocks = F;
    const dim3 blocks(x_blocks, y_blocks);

    // Run kernel on the GPU
    complex_mul_cuda_kernel<scalar_t><<<blocks, 1024>>>(
        x.data<scalar_t>(), y.data<scalar_t>(), out.data<scalar_t>(),
        N, F, C, H, W);
}

/**
Compile:
/usr/local/cuda/bin/nvcc -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/TH -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/local/ady/anaconda3/include/python3.6m -c complex_mul_kernel.cu -o complex_mul_kernel_stride_no_permute.out -std=c++11
nvcc -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/TH -I/local/ady/anaconda3/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/local/ady/anaconda3/include/python3.6m complex_mul_kernel_stride_no_permute.cu -o complex_mul_kernel_stride_no_permute.out -std=c++11
Segmentation fault
*/
int main(void)
{
    // auto dims = {128, 32, 16, 8, 2};
    at::Tensor x = at::randn({128, 32, 16, 8, 2});
    at::Tensor y = at::randn({128, 32, 16, 8, 2});
    at::Tensor out = at::zeros({128, 32, 16, 8, 2});
    complex_mul_stride_no_permute_cuda_pure<float>(x, y, out, 1024);
    return 0;
}