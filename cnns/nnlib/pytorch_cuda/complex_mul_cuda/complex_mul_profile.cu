#include <iostream>
#include <math.h>
#include "complex_mul.hpp"

int main(void)
{
    at::Tensor x = at::randn({128, 32, 16, 8, 2});
    at::Tensor y = at::randn({128, 32, 16, 8, 2});
    at::Tensor out = at::zeros({128, 32, 16, 8, 2});
    complex_mul_stride_no_permute_cuda_pure(x, y, out, 1024);
    return 0;
}