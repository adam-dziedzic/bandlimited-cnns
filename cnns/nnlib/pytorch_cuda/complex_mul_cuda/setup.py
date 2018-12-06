from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='complex_mul_cuda',
    ext_modules=[
        CUDAExtension('complex_mul_cuda', [
            'complex_mul_cuda.cpp',
            'complex_mul_kernel.cu',
            'complex_mul_kernel_stride.cu',
            'complex_mul_kernel_stride_no_permute.cu',
            'complex_mul_shared_log.cu',
            'complex_mul_deep.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

