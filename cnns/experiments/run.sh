#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../../ python memory_net.py --help

TIMESTAMP=$(date -d"${CURRENT}+${MINUTES}minutes" '+%F-%T.%N_%Z' | tr : - | tr . -)
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../../ nohup ~/anaconda3/bin/python memory_net.py --initbatchsize=256 --maxbatchsize=256 --startsize=32 --endsize=32 --workers=4 --device=cuda --limit_size=0 --num_epochs=300 --is_data_augmentation=True --conv_type=SPECTRAL_PARAM --optimizer=ADAM &> ${TIMESTAMP}-EXEC.log &


# memory efficient dense-net
TIMESTAMP=$(date -d"${CURRENT}+${MINUTES}minutes" '+%F-%T.%N_%Z' | tr : - | tr . -)
# conv_type=SPECTRAL_PARAM
conv_type=STANDARD
CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python demo.py --conv_type=${conv_type} --efficient False --data ../time-series-ml/cnns/pytorch_tutorials/data/cifar-10-batches-py --save save_spatial/ &> ${TIMESTAMP}-EXEC.log &
