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


PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=COMPRESS_INPUT_ONLY --epochs=300 --index_back=0 >> index_back_0_percent_input_compression_only-2018-09-18-15-49.txt 2>&1 &

PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=COMPRESS_INPUT_ONLY --epochs=300 --index_back=5 >> index_back_5_percent_input_compression_only-2018-09-18-15-49.txt 2>&1 &

PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=1 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=COMPRESS_INPUT_ONLY --epochs=300 --index_back=10 >> index_back_10_percent_input_compression_only-2018-09-18-15-49.txt 2>&1 &

PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=2 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=COMPRESS_INPUT_ONLY --epochs=300 --index_back=50 >> index_back_50_percent_input_compression_only-2018-09-18-15-49.txt 2>&1 &

PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=3 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=COMPRESS_INPUT_ONLY --epochs=300 --index_back=90 >> index_back_90_percent_input_compression_only-2018-09-18-15-49.txt 2>&1 &


PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=300 --index_back=0 --preserve_energy=80 >> index_back_0_preserve_energy_80_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;

PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=300 --index_back=0 --preserve_energy=70 >> index_back_0_preserve_energy_70_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;


PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=1 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=300 --index_back=0 --preserve_energy=90 >> index_back_0_preserve_energy_90_percent_full_fft1D-2018-09-22-12-26.txt 2>&1; PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=1 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=300 --index_back=0 --preserve_energy=95 >> index_back_0_preserve_energy_95_percent_full_fft1D-2018-09-22-12-26.txt 2>&1; PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=1 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=300 --index_back=0 --preserve_energy=99 >> index_back_0_preserve_energy_99_percent_full_fft1D-2018-09-22-12-26.txt 2>&1; PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=1 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=300 --index_back=0 --preserve_energy=50 >> index_back_0_preserve_energy_50_percent_full_fft1D-2018-09-22-12-26.txt 2>&1; PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=1 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=300 --index_back=0 --preserve_energy=10 >> index_back_0_preserve_energy_10_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;


PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/pytho0 fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=100 >> index_back_0_preserve_energy_100_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=99 >> index_back_0_preserve_energy_99_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=90 >> index_back_0_preserve_energy_90_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=95 >> index_back_0_preserve_energy_95_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=50 >> index_back_0_preserve_energy_50_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=30 >> index_back_0_preserve_energy_30_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=0 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=10 >> index_back_0_preserve_energy_10_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;

PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=2 nohup ~/anaconda3/bin/pytho0 fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=100 --network_type=SMALL >> small_index_back_0_preserve_energy_100_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=2 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=99  --network_type=SMALL >> small_index_back_0_preserve_energy_99_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=2 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=90  --network_type=SMALL >> small_index_back_0_preserve_energy_90_percent_full_fft1D-2018-09-22-12-26.small_txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=2 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=95  --network_type=SMALL >> small_index_back_0_preserve_energy_95_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=2 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=50  --network_type=SMALL >> small_index_back_0_preserve_energy_50_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=2 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=30  --network_type=SMALL >> small_index_back_0_preserve_energy_30_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;
PYTHONPATH=../../../ CUDA_VISIBLE_DEVICES=2 nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=debug --epochs=1 --min_batch_size=4096 --index_back=0 --preserve_energy=10  --network_type=SMALL >> small_index_back_0_preserve_energy_10_percent_full_fft1D-2018-09-22-12-26.txt 2>&1;


PYTHONPATH=../../../ nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=all --epochs=300 --min_batch_size=16 --index_back=0 --preserve_energy=90 --network_type=STANDARD --compress_type=BIG_COEFF >> big_coeff_index_back_0_preserve_energy_90_percent_full_fft1D-2018-09-22-12-26.txt 2>&1 &

PYTHONPATH=../../../ nohup ~/anaconda3/bin/python fcnn_fft.py --conv_type=FFT1D --datasets=all --epochs=300 --min_batch_size=16 --index_back=0 --preserve_energy=90 --network_type=STANDARD --compress_type=LOW_COEFF >> low_coeff_index_back_0_preserve_energy_90_percent_full_fft1D-2018-09-22-12-26.txt 2>&1 &


