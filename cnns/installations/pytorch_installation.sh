#!/usr/bin/env bash

nvcc --version
# conda install pytorch torchvision cuda90 -c pytorch --yes
conda install pytorch torchvision cuda92 -c pytorch --yes

python -c "import torch; print('torch version: ', torch.__version__)"
python -c "import torch; print('cuda version: ', torch.version.cuda)"
