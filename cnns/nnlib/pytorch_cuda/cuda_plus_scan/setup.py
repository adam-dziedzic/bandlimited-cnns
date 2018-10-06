from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='plus_scan_cuda',
    ext_modules=[
        CUDAExtension('plus_scan_cuda', [
            'plus_scan_cuda.cpp',
            'plus_scan_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
