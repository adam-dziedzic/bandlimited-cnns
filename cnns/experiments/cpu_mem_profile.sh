#!/usr/bin/env bash
PYTHONPATH=/home/adam/code/time-series-ml /home/adam/anaconda3/bin/python -m memory_profiler /home/adam/code/time-series-ml/cnns/nnlib/pytorch_timeseries/fcnn_fft.py --conv_type=FFT1D --epochs=1 --index_back=0 --preserve_energy=90 --datasets=debug --lr=0.001 --optimizer_type=ADAM --compress_type=STANDARD --min_batch_size=256
