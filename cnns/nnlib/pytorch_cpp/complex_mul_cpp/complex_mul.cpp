#include <torch/torch.h>

/*
    cat = torch.cat
    # ua = x.narrow(dim=-1, start=0, length=1)
    ua = x[..., :1]
    # ud = x.narrow(-1, 1, 1)
    ud = x[..., 1:]
    # va = y.narrow(-1, 0, 1)
    va = y[..., :1]
    # vb = y.narrow(-1, 1, 1)
    vb = y[..., 1:]
    ub = ua + ud
    uc = ud - ua
    vc = va + vb
    uavc = ua * vc
    # real part of the complex number
    # result_rel = add(uavc, mul(mul(ub, vb), -1))
    result_rel = uavc - ub * vb
    # imaginary part of the complex number
    result_im = uc * va + uavc
    # use the last dimension: dim=-1
    result = cat((result_rel, result_im), dim=-1)
    return result
*/

at::Tensor complex_mul_cpp(at::Tensor x, at::Tensor y) {
    at::Tensor ua = x.narrow(/*dim=*/-1, /*start=*/0, /*length=*/1);
    at::Tensor ud = x.narrow(/*dim=*/-1, /*start=*/1, /*length=*/1);

    at::Tensor va = y.narrow(/*dim=*/-1, /*start=*/0, /*length=*/1);
    at::Tensor vb = y.narrow(/*dim=*/-1, /*start=*/1, /*length=*/1);

    at::Tensor ub = ua + ud;
    at::Tensor uc = ud - ua;
    at::Tensor vc = va + vb;

    at::Tensor uavc = ua * vc;

    at::Tensor result_real = uavc - ub * vb;
    at::Tensor result_imag = uavc + uc * va;

    return at::cat(/*tensors=*/{result_real, result_imag}, /*dim=*/-1);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("complex_mul_cpp", &complex_mul_cpp,
  "cpp based multiplication of complex tensors");
}