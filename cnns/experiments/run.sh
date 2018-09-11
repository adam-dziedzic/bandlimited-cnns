#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../../ python memory_net.py --help

TIMESTAMP=$(date -d"${CURRENT}+${MINUTES}minutes" '+%F-%T.%N_%Z' | tr : - | tr . -)
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=../../ nohup ~/anaconda3/bin/python memory_net.py --initbatchsize=256 --maxbatchsize=256 --startsize=32 --endsize=32 --workers=4 --device=cuda --limit_size=0 --num_epochs=300 --is_data_augmentation=True --conv_type=SPECTRAL_PARAM --optimizer=ADAM &> ${TIMESTAMP}-EXEC.log &


# memory efficient dense-net
TIMESTAMP=$(date -d"${CURRENT}+${MINUTES}minutes" '+%F-%T.%N_%Z' | tr : - | tr . -)
# conv_type=SPECTRAL_PARAM
conv_type=STANDARD
CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python demo.py --conv_type=${conv_type} --efficient False --data ../time-series-ml/cnns/pytorch_tutorials/data/cifar-10-batches-py --save save_spatial/ &> ${TIMESTAMP}-EXEC.log &

PATH="~/anaconda3/bin:$PATH" CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup ~/anaconda3/bin/python fcnn_fft.py --index_back=5  >> log_index-back5.txt 2>&1 &


PATH="~/anaconda3/bin:$PATH" CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup ~/anaconda3/bin/python fcnn_fft.py --index_back=5  >> log_index-back5.txt 2>&1 &
PATH="~/anaconda3/bin:$PATH" CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup ~/anaconda3/bin/python fcnn_fft.py --index_back=2  >> log_index-back2.txt 2>&1 &
PATH="~/anaconda3/bin:$PATH" CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../ nohup ~/anaconda3/bin/python fcnn_fft.py --index_back=3  >> log_index-back3.txt 2>&1 &
PATH="~/anaconda3/bin:$PATH" CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../ nohup ~/anaconda3/bin/python fcnn_fft.py --index_back=4  >> log_index-back4.txt 2>&1 &

PATH="~/anaconda3/bin:$PATH" CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=SIMPLE_FFT_FOR_LOOP --epochs=300 --index_back=99 >> index_back_99_percent.txt 2>&1 &
