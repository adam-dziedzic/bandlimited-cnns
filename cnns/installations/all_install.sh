#!/usr/bin/env bash

#do this on your own first
# mkdir code
# cd code
# git clone https://github.com/adam-dziedzic/time-series-ml.git

sudo apt-get install emacs --yes

git config --global credential.helper cache

cd /home/${USER}/Downloads

code_path=/home/cc/code/time-series-ml/cnns/installations/

bash ${code_path}cuda_installation.sh
bash ${code_path}anaconda_installation.sh

source ~/.bashrc

sudo pip install typing

bash ${code_path}pytorch_from_source.sh
bash ${code_path}tensorflow_installation.sh

