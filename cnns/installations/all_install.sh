#!/usr/bin/env bash

# remove previous cuda (check if it exists)
sudo add-apt-repository -r cuda-repo-ubuntu1404_10.1.168-1_amd64.deb

#do this on your own first
# mkdir code
# cd code
# git clone https://github.com/adam-dziedzic/time-series-ml.git

# sudo apt-get install emacs --yes

git config --global credential.helper cache

cd /home/${USER}/Downloads

code_path=/home/cc/code/time-series-ml/cnns/installations/

bash ${code_path}cuda_installation.sh
bash ${code_path}anaconda_installation.sh

export PATH=~/anaconda3/bin:$PATH
source ~/.bashrc

sudo pip install typing

# bash ${code_path}pytorch_from_source.sh
bash ${code_path}pytorch_installation.sh
bash ${code_path}tensorflow_installation.sh
bash ${code_path}nvidia_apex.sh

pip install opencv-python


