#!/usr/bin/env bash

saved_dir =$(pwd)

cd /home/${USER}/Downloads

cd /home/${USER}/Downloads
git clone --recursive https://github.com/pytorch/vision.git

cd vision
python setup.py install

cd ${saved_dir}
