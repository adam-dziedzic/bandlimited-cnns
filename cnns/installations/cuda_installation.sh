#!/usr/bin/env bash

# on your own machine
# ssh cc@gpu
# mkdir -p Downloads
# exit;
# scp /home/${USER}/Downloads/chameleon/* cc@gpu:/home/cc/Downloads
# ssh cc@gpu

sudo apt-get install build-essential gcc-multilib dkms --yes

cd /home/${USER}/Downloads
sudo dpkg -i --force-overwrite cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update --yes
sudo apt-get install cuda --yes

sudo dpkg -i libcudnn7_7.1.4.18-1+cuda9.2_amd64.deb
sudo dpkg -i libcudnn7-dev_7.1.4.18-1+cuda9.2_amd64.deb

