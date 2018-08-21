#!/usr/bin/env bash

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
# the below command should upgrade:
# libcuda1-396 libxcursor-dev libxcursor1 libxnvctrl0 nvidia-396
# nvidia-396-dev nvidia-opencl-icd-396 nvidia-settings x11-common
sudo apt-get upgrade
reboot

