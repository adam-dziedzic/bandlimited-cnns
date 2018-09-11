#!/usr/bin/env bash

for epoch in 50 100 200 300; do
    echo ${epoch}
    for conv_type in STANDARD SIMPLE_FFT FFT1D; do
        echo ${conv_type}
        PYTHONPATH=../../../ ~/anaconda3/bin/python fcnn_fft.py --conv_type=${conv_type} --epochs=${epoch}
    done
done