#!/usr/bin/env bash

# PLAIN
PYTHON="/home/${USER}/anaconda3/envs/spsa/bin/python" # python environment
dataset=cifar10
num_classes=10
batch_size=1
target_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
target_arch=vanilla_resnet20
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
--spsa_samples 128 \
--spsa_iters 1 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt

# ADV TRAIN
PYTHON="/home/${USER}/anaconda3/envs/spsa/bin/python" # python environment
dataset=cifar10
num_classes=10
batch_size=1
target_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/adv_train.pth.tar"
target_arch=vanilla_resnet20
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
--spsa_samples 128 \
--spsa_iters 1 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
icml node
test_2020-04-02-23-50-41-449075481.txt

# PNI
PYTHON="/home/${USER}/anaconda3/envs/spsa/bin/python" # python environment
dataset=cifar10
num_classes=10
batch_size=1
target_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/pni.pth.tar"
target_arch=noise_resnet20_weight
source_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
source_arch=vanilla_resnet20
data_dir="/home/${USER}/data/pytorch/${dataset}/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup $PYTHON blackbox.py \
--ngpu 1 \
--batch_size ${batch_size} \
--num_classes ${num_classes} \
--data_dir ${data_dir} \
--attack_type 'spsa' \
--epsilon 0 \
--epsilons 0.031 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
--iter 2048 \
--cw_conf 20 \
--spsa_samples 128 \
--spsa_iters 1 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[3] 118091
(spsa) cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/blackbox_attacks$ echo test_${timestamp}.txt
test_2020-04-03-02-16-26-802238972.txt

# ROBUST NET
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
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup $PYTHON blackbox.py \
--ngpu 1 \
--batch_size ${batch_size} \
--num_classes ${num_classes} \
--data_dir ${data_dir} \
--attack_type 'spsa' \
--epsilon 0 \
--epsilons 0.031 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
--iter 2048 \
--cw_conf 20 \
--spsa_samples 128 \
--spsa_iters 1 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 117476
(spsa) cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/blackbox_attacks$ echo test_${timestamp}.txt
test_2020-04-03-02-01-54-270914440.txt

# ROBUST + ADV
PYTHON="/home/${USER}/anaconda3/envs/spsa/bin/python" # python environment
dataset=cifar10
num_classes=10
batch_size=1
target_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_01_adv_train.pth.tar"
target_arch=noise_resnet20_robust_01
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
--spsa_samples 128 \
--spsa_iters 1 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 12935
(spsa) ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/blackbox_attacks$ echo test_${timestamp}.txt
test_2020-04-02-21-17-38-625335105.txt

# PLAIN
PYTHON="/home/${USER}/anaconda3/envs/spsa-gpu3/bin/python" # python environment
dataset=cifar10
num_classes=10
batch_size=1
target_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
target_arch=vanilla_resnet20
source_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
source_arch=vanilla_resnet20
data_dir="/home/${USER}/data/pytorch/${dataset}/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=../../../../ nohup $PYTHON blackbox.py \
--ngpu 4 \
--batch_size ${batch_size} \
--num_classes ${num_classes} \
--data_dir ${data_dir} \
--attack_type 'spsa' \
--epsilon 0 \
--epsilons 0.031 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
--iter 100 \
--cw_conf 20 \
--spsa_samples 512 \
--spsa_iters 4 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &


# PLAIN
PYTHON="/home/${USER}/anaconda3/envs/spsa/bin/python" # python environment
dataset=cifar10
num_classes=10
batch_size=1
target_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
target_arch=vanilla_resnet20
source_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
source_arch=vanilla_resnet20
data_dir="/home/${USER}/data/pytorch/${dataset}/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=../../../../ nohup $PYTHON blackbox.py \
--ngpu 0 \
--no_cuda \
--batch_size ${batch_size} \
--num_classes ${num_classes} \
--data_dir ${data_dir} \
--attack_type 'spsa' \
--epsilon 0 \
--epsilons 0.031 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
--iter 100 \
--cw_conf 20 \
--spsa_samples 8192 \
--spsa_iters 1 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/envs/spsa/bin/python" # python environment
dataset=cifar10
num_classes=10
batch_size=1
target_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
target_arch=vanilla_resnet20
source_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar"
source_arch=vanilla_resnet20
data_dir="/home/${USER}/data/pytorch/${dataset}/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup $PYTHON blackbox.py \
--ngpu 1 \
--tf_cpu \
--batch_size ${batch_size} \
--num_classes ${num_classes} \
--data_dir ${data_dir} \
--attack_type 'spsa' \
--epsilon 0 \
--epsilons 0.031 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
--iter 100 \
--cw_conf 20 \
--spsa_samples 8192 \
--spsa_iters 1 \
--save_path './save/' \
--target_model ${target_model} \
--target_arch ${target_arch} \
--source_model ${source_model} \
--source_arch ${source_arch} \
--manual_seed 31 \
>> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
