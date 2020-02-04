#!/usr/bin/env bash

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --adv_eval --epoch_delay 5 \
    --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt