#!/usr/bin/env bash

with_cuda=${1:-"FALSE"}

if [ "${with_cuda}" == "FALSE" ]; then
    echo "installation without CUDA support"
    export NO_CUDA=1
    export NO_CUDNN=1
    export NO_MKLDNN=1
    export NO_NNPACK=1
    export NO_DISTRIBUTED=1
    export NO_SYSTEM_NCCL=1
fi

export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install --yes numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install --yes -c mingfeima mkldnn
conda install --yes libgcc
conda install --yes -c caffe2 caffe
conda install --yes -c sci-bots nanopb

if [ "${with_cuda}" == "TRUE" ]; then
    # Add LAPACK support for the GPU
    conda install -c pytorch magma-cuda90 # or magma-cuda90 if CUDA 9, magma-cuda80 if CUDA 8
fi

python setup.py install


