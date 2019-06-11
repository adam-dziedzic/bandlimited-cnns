from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='band_cuda',
    ext_modules=[
        CUDAExtension('band_cuda', [
            'band_cuda.cpp',
            'band_cuda_kernel.cu',
            'complex_mul.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
