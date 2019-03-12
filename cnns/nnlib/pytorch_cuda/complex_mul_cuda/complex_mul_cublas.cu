//  Band-limited CNNs
//  Copyright (c) 2019. Adam Dziedzic
//  Licensed under The Apache License [see LICENSE for details]
//  Written by Adam Dziedzic

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#include <ATen/ATen.h>

#include <cuda.h>
#include <string>

namespace {

void complex_mul_cublas(
    at::Tensor x,
    at::Tensor y,
    at::Tensor out) {

    const auto H = x.size(0);
    const auto W = x.size(1);
    const auto N = x.size(2);  // batch_size
    const auto C = x.size(3);  // number of channels
    const auto F = y.size(2);  // filter_bank_size

    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw "CUBLAS initialization failed\n";
    }

    cublasOperation_t transa = CUBLAS_OP_N; // 	the non-transpose operation is selected
    cublasOperation_t transb = CUBLAS_OP_N;

    cuComplex alpha = make_cuComplex(1.0, 0.0);

    cublasStatus_t stat;
    stat = cublasStatus_t cublasCgemm(handle,
                           transa, transb,
                           int m, int n, int k,
                           &alpha,
                           const cuComplex       *A, int lda,
                           const cuComplex       *B, int ldb,
                           const cuComplex       *beta,
                           cuComplex       *C, int ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        throw "CUBLAS cGemm failed\n";
    }
}

}

