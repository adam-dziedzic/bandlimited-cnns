#!/usr/bin/env bash

sudo apt-get install emacs --yes

cd /home/${USER}/Downloads

bash cuda_installation.sh
bash anaconda_installation.sh
bash pytorch_from_source.sh
bash tensorflow_installation.sh

