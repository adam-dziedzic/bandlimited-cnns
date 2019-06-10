#include <torch/torch.h>

#include <vector>

// CUDA declarations
std::vector<at::Tensor> band_cuda_forward(
    at::Tensor input,
    at::Tensor weights);

std::vector<at::Tensor> band_cuda_backward(
    at::Tensor grad);

void complex_mul_cuda(
    at::Tensor x, at::Tensor y, at::Tensor out, int threads = 1024);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor band_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

    at::Tensor xfft = input.rfft(
            /*signal_ndim*/2, /*normalized*/false, /*onesided*/true);
    at::Tensor yfft = input.rfft(
            /*signal_ndim*/2, /*normalized*/false, /*onesided*/true);


    return xfft;
}

std::vector<at::Tensor> band_backward(
    at::Tensor grad) {
  CHECK_INPUT(grad);

  return std::vector<at::Tensor>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &band_forward, "band forward");
  m.def("backward", &band_backward, "band backward");
}