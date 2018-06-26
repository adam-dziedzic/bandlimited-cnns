echo "Download libraries from ady ryerson for GPU and CUDA"

init_dir=`pwd`

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
su ${main_user}
cd ${current_dir}
. /home/${main_user}/torch/install/bin/torch-activate
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
