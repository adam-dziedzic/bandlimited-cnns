#!/usr/bin/env bash

PYTHON="/home/${USER}/anaconda3/envs/spsa/bin/python" # python environment
dataset=cifar10
num_classes=10
batch_size=1
target_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_0.2.pth.tar"
target_arch=noise_resnet20_robust_02
source_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
source_arch=vanilla_resnet20
data_dir="/home/${USER}/data/pytorch/${dataset}/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup $PYTHON blackbox.py \
--ngpu 1 \
--batch_size ${batch_size} \
--num_classes ${num_classes} \
--data_dir ${data_dir} \
--attack_type 'spsa' \
--epsilon 0 \
--epsilons 0.031 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
--iter 2048 \
--cw_conf 20 \
--spsa_samples 2048 \
--spsa_iters 1 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 27192
(spsa) ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/blackbox_attacks$ echo test_${timestamp}.txt
test_2020-04-02-17-37-21-188419729.txt




