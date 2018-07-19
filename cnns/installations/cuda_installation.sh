#!/usr/bin/env bash

sudo apt-get install build-essential gcc-multilib dkms

sudo dpkg -i --force-overwrite cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda --yes

sudo dpkg -i libcudnn7_7.1.4.18-1+cuda9.2_amd64.deb
sudo dpkg -i libcudnn7-dev_7.1.4.18-1+cuda9.2_amd64.deb

