from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='complex_mul_cpp',
      ext_modules=[CppExtension('complex_mul_cpp', ['complex_mul.cpp'])],
      cmdclass={'build_ext': BuildExtension})
