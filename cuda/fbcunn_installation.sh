echo "Download libraries from ady ryerson for GPU and CUDA"

init_dir=`pwd`

# echo "install anaconda"
# yes | wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
# source ~/.bashrc
# y | conda install -c anaconda thrift
# sudo su
# echo 'export PATH="/home/cc/anaconda3/bin:$PATH"' >> ~/.bashrc
# su cc

echo "install emacs"
sudo apt-get -y install emacs

echo "install htop"
sudo apt-get -y install htop

sudo apt-get -y install scons

main_user=`$USER`
# install CUDA
sudo apt-get -y install build-essential
sudo apt-get update
sudo apt-get -y install linux-generic
sudo dpkg -i ~/Downloads/cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

echo "export PATH=/usr/local/cuda/bin/:\$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:\$LD_LIBRARY_PATH; " >>~/.bashrc && source ~/.bashrc

echo "show cuda version"
cat /usr/local/cuda/version.txt

echo "install CuDNN from NVidia"
sudo dpkg -i ~/Downloads/libcudnn7_7.1.4.18-1+cuda9.2_amd64.deb
sudo dpkg -i ~/Downloads/libcudnn7-dev_7.1.4.18-1+cuda9.2_amd64.deb
sudo dpkg -i ~/Downloads/libcudnn7-doc_7.1.4.18-1+cuda9.2_amd64.deb

echo "install libqt4-dev"
sudo apt-get -y install libqt4-dev

echo "install torch dependencies"
curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash -e

echo "install cmake"
sudo apt-get -y install cmake

echo "Install Torch distro in a local folder"
current_dir=`pwd`
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
sudo su
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
bash install-deps
yes | bash ./install.sh
ldconfig
su ${main_user}
cd ${current_dir}
. /home/${main_user}/torch/install/bin/torch-activate
source torch/install/bin/torch-activate
echo ". /home/${USER}/torch/install/bin/torch-activate" >> ~/.bashrc

source ~/.bashrc

echo "install boost"
# https://coderwall.com/p/0atfug/installing-boost-1-55-from-source-on-ubuntu-12-04
sudo apt-get -y install build-essential g++ python-dev autotools-dev libicu-dev build-essential libbz2-dev
wget https://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz
tar -xf boost_1_55_0.tar.gz
dir=`pwd`
cd boost_1_55_0
./bootstrap.sh --prefix=/usr/local
# find the max number of physical cores
n=`cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $NF}'`
# install boost in parallel
sudo ./b2 --with=all -j $n install 
# Assumes you have /usr/local/lib setup already. if not, you can add it to your LD LIBRARY PATH
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/local.conf'
cd ${dir}
# Reset the ldconfig:
sudo ldconfig

echo "install folly, fbthrift, th++, fblualib"
sudo apt-get -y install libpthread-stubs0-dev
sudo apt-get -y install libpthread-workqueue-dev

git clone https://github.com/google/double-conversion
cd double-conversion
sudo scons install
cd ${init_dir}

sudo apt-get install libgflags-dev

#!/bin/bash -e
#
#  Copyright (c) 2014, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
init_dir=`pwd`
echo
echo This script will install fblualib and all its dependencies.
echo It has been tested on Ubuntu 13.10 and Ubuntu 14.04, Linux x86_64.
echo

set -e
set -x

if [[ $(arch) != 'x86_64' ]]; then
    echo "x86_64 required" >&2
    exit 1
fi

issue=$(cat /etc/issue)
extra_packages=
if [[ $issue =~ ^Ubuntu\ 13\.10 ]]; then
    :
elif [[ $issue =~ ^Ubuntu\ 14 ]]; then
    extra_packages=libiberty-dev
elif [[ $issue =~ ^Ubuntu\ 16 ]]; then
    extra_packages=libiberty-dev
    echo "Trying to install on ubuntu 16.04"
    echo
else
    echo "Ubuntu 13.10 or 14.* required" >&2
    exit 1
fi

dir=$(mktemp --tmpdir -d fblualib-build.XXXXXX)

echo Working in $dir
echo
cd $dir

echo Installing required packages
echo
sudo apt-get install -y \
    git \
    curl \
    wget \
    g++ \
    cmake \
    libboost-all-dev \
    automake \
    autoconf \
    autoconf-archive \
    libtool \
    libevent-dev \
    libdouble-conversion-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    liblz4-dev \
    liblzma-dev \
    libsnappy-dev \
    make \
    zlib1g-dev \
    binutils-dev \
    libjemalloc-dev \
    libssl-dev \
    $extra_packages \
    flex \
    bison \
    libkrb5-dev \
    libsasl2-dev \
    libnuma-dev \
    pkg-config \
    libssl-dev \
    libedit-dev \
    libmatio-dev \
    libpython-dev \
    libpython3-dev \
    python-numpy \
    libunwind8-dev \
    libelf-dev \
    libdwarf-dev

echo
echo Cloning repositories
echo
# git clone -b v0.35.0  --depth 1 https://github.com/facebook/folly.git
# git clone -b v0.24.0  --depth 1 https://github.com/facebook/fbthrift.git
# git clone https://github.com/facebook/thpp
git clone https://github.com/soumith/fblualib

echo
echo Building folly
echo

# cd $dir/folly/folly
# autoreconf -ivf
# ./configure
# make
# sudo make install
# sudo ldconfig # reload the lib paths after freshly installed folly. fbthrift needs it.

dir=`pwd`
wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz && \
tar zxf release-1.8.0.tar.gz && \
rm -f release-1.8.0.tar.gz && \
cd googletest-release-1.8.0 && \
cmake configure . && \
make && \
sudo make install
sudo ldconfig # reload the lib paths after freshly installed folly. fbthrift needs it.  
cd ${dir}

init_dir=`pwd`
wget https://github.com/facebook/folly/archive/v2018.06.18.00.tar.gz
tar -xf v2018.06.18.00.tar.gz
cd folly-2018.06.18.00
mkdir _build && cd _build
cmake configure ..
make -j $(nproc)
sudo make install
sudo ldconfig
cd ${init_dir}

echo
echo Building fbthrift
echo

echo "mstch"
git clone https://github.com/no1msd/mstch
cd mstch
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
cd ${init_dir}

echo "wangle"
git clone https://github.com/facebook/wangle
cd wangle/wangle
cmake .
make
ctest
sudo make install
sudo ldconfig
cd ${init_dir}

echo "zstd"
init_dir=`pwd`
wget https://github.com/facebook/zstd/archive/v1.3.4.tar.gz
tar -xf v1.3.4.tar.gz 
vm v1.3.4.tar.gz
cd zstd-1.3.4
make
sudo make install
sudo ldconfig
cd ${init_dir}

echo "thrift"
# cd $dir/fbthrift/thrift
# autoreconf -ivf
# ./configure
# make
# sudo make install

dir=`pwd`
git clone https://github.com/facebook/fbthrift
cd fbthrift
# echo "set(CMAKE_CXX_FLAGS "-fPIC")" >> CMakeLists.txt 
# echo "set(CMAKE_C_FLAGS "-fPIC")" >> CMakeLists.txt
# https://stackoverflow.com/questions/9533679/how-to-insert-a-text-at-the-beginning-of-a-file
# https://github.com/facebook/fbthrift/issues/249
sed -i '6s/^/set(CMAKE_CXX_FLAGS "-fPIC")\n/' CMakeLists.txt
sed -i '7s/^/set(CMAKE_C_FLAGS "-fPIC")\n/' CMakeLists.txt

cd build
cmake .. # Add -DOPENSSL_ROOT_DIR for macOS. Usually in /usr/local/ssl
# make # or make -j $(nproc), or make install.
make -j $(nproc)
sudo make install
sudo ldconfig
cd ${dir}

# more problems with thrift
# in ~/fbthrift/thrift/compiler/py$
init_dir=`pwd`
cd fbthrift/thrift/compiler/py
sudo g++ -I /usr/include/python2.7 -I /home/${USER}/fbthrift -std=c++14 -fpic -shared -o frontend.so compiler.cc -lboost_python -lpython2.7 -L/build/lib -lcompiler_base -lcompiler_ast -lboost_system -lboost_filesystem -lssl -lcrypto
cd ${init_dir}

init_dir=`pwd`
# it has to be python 2.7
mkdir tmp
cp -a fbthrift/thrift/compiler/py tmp/
cd tmp/py
sudo python setup.py install
cd ${init_dir}

# added the two "set command" in the libs_only section of the fbthrift CMakeList.txt
# set(CMAKE_CXX_FLAGS "-fPIC")
# set(CMAKE_C_FLAGS "-fPIC")

echo
echo 'Installing TH++'
echo

init_dir=`pwd`
cd ~/
source ~/torch/install/bin/activate-tource
git clone https://github.com/facebook/thpp
cd thpp/thpp
# git reset --hard ce40d2704267f33098a595f9f9f612c6e06e330e
git reset --hard 2b285e3e89bdd27bebc5d13135b13943b028ac89
sed -i '6s/^/set(CMAKE_CXX_STANDARD 14)\n/' CMakeLists.txt
./build.sh
cd ${init_dir}

echo
echo 'Installing FBLuaLib'
echo


cd $dir/fblualib/fblualib
sudo ./build.sh

echo
echo 'All done!'
echo

cd ${init_dir}

# install fbcunn

git clone https://github.com/torch/nn && ( cd nn && git checkout getParamsByDevice && luarocks make rocks/nn-scm-1.rockspec )

git clone https://github.com/facebook/fbtorch.git && ( cd fbtorch && luarocks make rocks/fbtorch-scm-1.rockspec )

git clone https://github.com/facebook/fbnn.git && ( cd fbnn && luarocks make rocks/fbnn-scm-1.rockspec )

# go get a coffee
git clone https://github.com/facebook/fbcunn.git && ( cd fbcunn && luarocks make rocks/fbcunn-scm-1.rockspec )

