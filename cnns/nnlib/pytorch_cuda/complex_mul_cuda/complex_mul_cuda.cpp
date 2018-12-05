#include <torch/torch.h>

void complex_mul_cuda(at::Tensor x, at::Tensor y, at::Tensor out);
void complex_mul_stride_cuda(at::Tensor x, at::Tensor y, at::Tensor out);
void complex_mul_stride_no_permute_cuda(
    at::Tensor x, at::Tensor y, at::Tensor out, int threads = 1024);

/* The implementation used the shared memory, synchronization and the summation
along the channels in the O(log N) complexity. */
void complex_mul_shared_log_cuda(at::Tensor x, at::Tensor y, at::Tensor out);

// for profiling
void complex_mul_stride_no_permute_cuda_pure(
    at::Tensor x, at::Tensor y, at::Tensor out, int threads = 1024);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void check_input(at::Tensor x, at::Tensor y, at::Tensor out) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(out)
}

void complex_mul(at::Tensor x, at::Tensor y, at::Tensor out) {
    check_input(x, y, out);
    complex_mul_cuda(x, y, out);
}

void complex_mul_stride(at::Tensor x, at::Tensor y, at::Tensor out) {
    check_input(x, y, out);
    complex_mul_stride_cuda(x, y, out);
}

void complex_mul_stride_no_permute(at::Tensor x, at::Tensor y, at::Tensor out,
    int threads=1024) {
    check_input(x, y, out);
    complex_mul_stride_no_permute_cuda(x, y, out, threads);
}

void complex_mul_shared_log(at::Tensor x, at::Tensor y, at::Tensor out) {
    check_input(x, y, out);
    complex_mul_shared_log_cuda(x, y, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("complex_mul", &complex_mul, "CUDA based multiplication of complex tensors.");
  m.def("complex_mul_stride", &complex_mul_stride, "CUDA based multiplication of complex tensors with stride.");
  m.def("complex_mul_stride_no_permute", &complex_mul_stride_no_permute, "CUDA based multiplication of complex tensors with stride and no permutation of the tensor.");
  m.def("complex_mul_shared_log", &complex_mul_shared_log, "CUDA based multiplication of complex tensors with shared memory, synchronization and O(log N) summation across channels.");
}