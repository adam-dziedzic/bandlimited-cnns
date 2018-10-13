#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 GPU_DEBUG=0 PYTHONPATH=/home/${USER}/code/time-series-ml /home/${USER}/anaconda3/bin/python /home/${USER}/code/time-series-ml/cnns/nnlib/pytorch_timeseries/fcnn_fft.py --conv_type=FFT1D --epochs=1 --index_back=0 --preserve_energy=90 --datasets=debug --is_debug=True --lr=0.001 --optimizer_type=ADAM --compress_type=STANDARD --sample_count_limit=256 --min_batch_size=256
