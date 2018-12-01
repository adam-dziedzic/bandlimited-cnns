#include <torch/torch.h>

// CUDA forward declarations

void complex_mul_cuda(at::Tensor x, at::Tensor y, at::Tensor out);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void complex_mul(at::Tensor x, at::Tensor y, at::Tensor out) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(out)

    complex_mul_cuda(x, y, out);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("complex_mul", &complex_mul,
  "cuda based multiplication of complex tensors");
}