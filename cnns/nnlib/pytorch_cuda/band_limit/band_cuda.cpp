#include <torch/torch.h>

#include <vector>

// CUDA declarations
std::vector <at::Tensor> band_cuda_forward(
        at::Tensor input,
        at::Tensor weights);

std::vector <at::Tensor> band_cuda_backward(
        at::Tensor grad);

void complex_mul_cuda(
        at::Tensor x, at::Tensor y, at::Tensor out, int threads = 1024);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> band_forward(
        at::Tensor input,
        at::Tensor weights) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);

    at::Tensor xfft = input.rfft(
            /*signal_ndim*/2, /*normalized*/false, /*onesided*/true);
    at::Tensor yfft = weights.rfft(
            /*signal_ndim*/2, /*normalized*/false, /*onesided*/true);
    /* Conjugate */
    yfft.narrow(/*dim*/4, /*start*/1, /*length*/1).mul_(-1);
    at::Tensor outfft = torch::zeros_like(xfft);
    complex_mul_cuda(/*x*/xfft, /*y*/yfft, /*out*/outfft);
    at::Tensor out = outfft.irfft(
            /*signal_ndim*/2, /*normalized*/false, /*onesided*/true);
    std::vector<at::Tensor> result;
    result.push_back(out);
    result.push_back(xfft);
    result.push_back(yfft);
    return result;
}

std::vector <at::Tensor> band_backward(
        at::Tensor grad, at::Tensor xfft, at::Tensor yfft) {
    CHECK_INPUT(grad);

    return std::vector<at::Tensor>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("forward", &band_forward, "band forward");
m.def("backward", &band_backward, "band backward");
}