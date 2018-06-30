#!/usr/bin/env bash
# install pytorch framework from the source code

with_cuda=${1:-"FALSE"} # set to true to install for GPU
conda_env=${2:-"base"} # we use pytorch mainly

if [ "${with_cuda}" == "TRUE" ]; then
    echo "installation with CUDA support "
elif [ "${with_cuda}" == "FALSE" ]; then
    echo "installation without CUDA support"
    export NO_CUDA=1
    export NO_CUDNN=1
    export NO_MKLDNN=1
    export NO_NNPACK=1
    export NO_DISTRIBUTED=1
    export NO_SYSTEM_NCCL=1
fi

echo "conda environment for intallation: "${conda_env}
source activate ${conda_env}

export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install -n ${conda_env} --yes numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -n ${conda_env} --yes -c mingfeima mkldnn
conda install -n ${conda_env} --yes libgcc
conda install -n ${conda_env} --yes -c caffe2 caffe
conda install -n ${conda_env} --yes -c sci-bots nanopb

if [ "${with_cuda}" == "TRUE" ]; then
    # Add LAPACK support for the GPU
    conda install -n ${conda_env} --yes c pytorch magma-cuda90 # or magma-cuda90 if CUDA 9, magma-cuda80 if CUDA 8
fi

mkdir -p code/pytorch/
cd code/pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py install


