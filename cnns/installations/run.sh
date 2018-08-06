#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../../ python memory_net.py --help

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../../ nohup ~/anaconda3/bin/python memory_net.py --initbatchsize=256 --maxbatchsize=256 --startsize=32 --endsize=32 --workers=4 --device=cuda --limit_size=0 --num_epochs=300 --is_data_augmentation=True --conv_type=SPECTRAL_PARAM --optimizer=ADAM &> log2018-08-06-09-33.log &
