#!/usr/bin/env bash
# first install the distributed version

conda install pytorch torchvision cuda92 -c pytorch --yes
# conda install pytorch torchvision cuda90 -c pytorch --yes

# install pytorch framework from the source code

with_cuda=${1:-"TRUE"} # set to true to install for GPU
conda_env=${2:-"base"} # we use pytorch mainly

if [ "${with_cuda}" == "TRUE" ]; then
    echo "installation with CUDA support: it requires cuda and libdnn installed from NVIDIA!!!"
    echo "read: https://github.com/pytorch/pytorch"
elif [ "${with_cuda}" == "FALSE" ]; then
    echo "installation without CUDA support"
    export NO_CUDA=1
    export NO_CUDNN=1
    export NO_MKLDNN=1
    export NO_NNPACK=1
    export NO_DISTRIBUTED=1
    export NO_SYSTEM_NCCL=1
fi

echo "prepare python/conda for the installation"
echo "conda environment for installation: "${conda_env}
source activate ${conda_env}

export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install -n ${conda_env} --yes libgcc numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -n ${conda_env} --yes -c mingfeima mkldnn
conda install -n ${conda_env} --yes -c caffe2 caffe

if [ "${with_cuda}" == "TRUE" ]; then
    # Add LAPACK support for the GPU
    conda install -n ${conda_env} --yes -c pytorch magma-cuda90 # or magma-cuda90 if CUDA 9, magma-cuda80 if CUDA 8
fi

echo "clone or update the source code and install it"
echo "go to the home directory"
cd
echo `pwd`
code_dir="code/pytorch/pytorch"
# -L denots a symbolic link
if [[ -d ${code_dir} ]] || [[ -L ${code_dir} ]]; then
    echo "The pytorch was already cloned from github"
    cd ${code_dir}
    git pull
else
    echo "Clone pytorch from github"
    init_dir="code/pytorch/"
    mkdir -p ${init_dir}
    cd ${init_dir}
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
fi

python setup.py install


