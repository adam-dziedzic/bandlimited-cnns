#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main.py --adam_beta2=0.999 --compress_type='STANDARD' --compress_rates 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar10' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=0 --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='STANDARD' --min_batch_size=32 --model_path='2019-01-11-02-21-05-406721-dataset-cifar10-preserve-energy-100.0-test-accuracy-92.23-51.5-real-compression.model' --momentum=0.9 --network_type='ResNet18' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies=100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='TRUE' --weight_decay=0.0001 --workers=6  --precision_type=FP32 --only_train=FALSE >> 2019-01-19-22-08-cifar10-fft2d-energy100-pytorch-model-51.5-compress.txt 2>&1 &
CUDA_PID=$!
wait ${CUDA_PID}

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main.py --adam_beta2=0.999 --compress_type='STANDARD' --compress_rates 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar10' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=0 --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='STANDARD' --min_batch_size=32 --model_path='2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model' --momentum=0.9 --network_type='ResNet18' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies=100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='TRUE' --weight_decay=0.0001 --workers=6  --precision_type=FP32 --only_train=FALSE >> 2019-01-19-22-08-cifar10-fft2d-energy100-pytorch-model-0-compress.txt 2>&1 &
CUDA_PID=$!
wait ${CUDA_PID}

# 2019-01-08-07-18-52-980249-dataset-cifar10-preserve-energy-100.0-test-accuracy-92.82-compress-29.26.model

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main.py --adam_beta2=0.999 --compress_type='STANDARD' --compress_rates 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar100' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=0  --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='STANDARD' --min_batch_size=32 --model_path='2019-01-19-21-38-38-075445-dataset-cifar100-preserve-energy-100.0-test-accuracy-63.9-compress-rate-76-real-compression-77.83.model' --momentum=0.9 --network_type='DenseNetCifar' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies=100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='TRUE' --weight_decay=0.0001 --workers=6  --precision_type=FP32 --only_train=FALSE >> 2019-01-19-22-08-cifar10-fft2d-energy100-pytorch-model-0-compress.txt 2>&1 &
CUDA_PID=$!
wait ${CUDA_PID}

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main.py --adam_beta2=0.999 --compress_type='STANDARD' --compress_rates 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar10' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=0 --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='STANDARD' --min_batch_size=32 --model_path='2019-01-08-02-48-26-558883-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.07-compress-8.32.model' --momentum=0.9 --network_type='ResNet18' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies=100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='TRUE' --weight_decay=0.0001 --workers=6  --precision_type=FP32 --only_train=FALSE >> 2019-01-19-22-08-cifar10-fft2d-energy100-pytorch-model-0-compress-8.32.txt 2>&1 &
CUDA_PID=$!
wait ${CUDA_PID}

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main.py --adam_beta2=0.999 --compress_type='STANDARD' --compress_rates 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar10' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=0 --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='STANDARD' --min_batch_size=32 --model_path='2019-01-08-07-18-52-980249-dataset-cifar10-preserve-energy-100.0-test-accuracy-92.82-compress-29.26.model' --momentum=0.9 --network_type='ResNet18' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies=100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='TRUE' --weight_decay=0.0001 --workers=6  --precision_type=FP32 --only_train=FALSE >> 2019-01-19-22-08-cifar10-fft2d-energy100-pytorch-model-0-compress-29.26.txt 2>&1 &
CUDA_PID=$!
wait ${CUDA_PID}

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main.py --adam_beta2=0.999 --compress_type='STANDARD' --compress_rates 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar10' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=0 --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='STANDARD' --min_batch_size=32 --model_path='2019-01-08-12-54-42-586365-dataset-cifar10-preserve-energy-100.0-test-accuracy-84.97-index-back-55.model' --momentum=0.9 --network_type='ResNet18' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies=100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='TRUE' --weight_decay=0.0001 --workers=6  --precision_type=FP32 --only_train=FALSE >> 2019-01-19-22-08-cifar10-fft2d-energy100-pytorch-model-0-compress-90.txt 2>&1 &
CUDA_PID=$!
wait ${CUDA_PID}






