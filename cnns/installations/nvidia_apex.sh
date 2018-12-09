#!/usr/bin/env bash

"""
NVIDIA-maintained utilities to streamline mixed precision and distributed
training in Pytorch. Some of the code here will be included in upstream Pytorch
eventually. The intention of Apex is to make up-to-date utilities available to
users as quickly as possible.
"""

saved_dir =$(pwd)

cd /home/${USER}/Downloads

cd /home/${USER}/Downloads
git clone --recursive https://github.com/NVIDIA/apex

cd apex
~/anaconda3/bin/python setup.py install


cd ${saved_dir}
