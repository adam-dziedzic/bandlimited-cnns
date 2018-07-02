#!/usr/bin/env bash

# first try the local version:
sudo dpkg -i --force-overwrite cuda-repo-ubuntu1710-9-2-local_9.2.88-1_amd64.deb

# otherwise
sudo dpkg -i cuda-repo-ubuntu1710_9.2.88-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda


# use: sudo dpkg -i --force-overwrite : to install libcudnn

sudo dpkg -i --force-overwrite libcudnn7_7.1.4.18-1+cuda9.2_amd64.deb
sudo dpkg -i --force-overwrite libcudnn7-dev_7.1.4.18-1+cuda9.2_amd64.deb
