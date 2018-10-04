from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv1D_cuda',
    ext_modules=[
        CUDAExtension('conv1D_cuda', [
            'conv_cuda.cpp',
            'conv_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

