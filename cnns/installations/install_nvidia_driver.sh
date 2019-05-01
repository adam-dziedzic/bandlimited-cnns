#!/usr/bin/env bash

# check drivers

lshw -numeric -C display
lspci -vnn | grep VGA
lspci -vnn | grep VGA

sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get install linux-headers-$(uname -r)


sudo apt-get install build-essential gcc-multilib dkm
sudo emacs /etc/modprobe.d/blacklist-nouveau.conf

paste:
blacklist nouveau
options nouveau modeset=0

sudo update-initramfs -u

sudo service lightdm stop

wget us.download.nvidia.com/tesla/410.104/NVIDIA-Linux-x86_64-410.104.run
chmod +x us.download.nvidia.com/tesla/410.104/NVIDIA-Linux-x86_64-410.104.run

sudo ./NVIDIA-Linux-x86_64-410.104.run --dkms -s

sudo reboot

nvidia-smi

