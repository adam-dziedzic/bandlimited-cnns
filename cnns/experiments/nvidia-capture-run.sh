#!/usr/bin/env bash

for compress_rate in 0 25 50 75; do
    for data_type in "fp16" "fp32"; do
        bash nvidia-capture.sh ${compress_rate} 1 32 ${data_type} "TRUE"
    done;
done;

