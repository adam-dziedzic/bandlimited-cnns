#!/bin/bash
#
#SBATCH --mail-user=ady@uchicago.edu
#SBATCH --mail-type=ALL

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 0.04 --net_mode '0-0' --attack_iters 300 --batch_size 1024 --limit_batch_number 0 --c 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
