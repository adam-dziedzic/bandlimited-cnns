#!/usr/bin/env bash

for data_type in "fp32" "fp16"; do
    for compress_rate in 0 25 50 75; do

        if [ "${type}" == "fp16" ]; then
            program_name="main_fp16_optimizer.py"
            tensor_type="FLOAT16"
            precision="FP16"
        else
            program_name="main.py"
            tensor_type="FLOAT32"
            precision="FP32"
        fi
        mem_test="FALSE"

        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../ nohup /home/${USER}/anaconda3/bin/python3.6 /home/${USER}/code/time-series-ml/cnns/nnlib/pytorch_experiments/${program_name} --adam_beta2=0.999 --compress_type='STANDARD' --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar10' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=0 --compress_rates ${compress_rate} --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size='FALSE' --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test=${mem_test} --memory_size=25 --memory_type='STANDARD' --min_batch_size=${size} --model_path='no_model' --momentum=0.9 --network_type='ResNet18' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies 100 --sample_count_limit=${size} --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type=${tensor_type} --test_batch_size=${size} --use_cuda='TRUE' --visualize='FALSE' --weight_decay=0.0005 --workers=6 --precision_type=${precision} --only_train='FALSE' >> ${TIMESTAMP}-cifar10-fft2d-energy100-pytorch-adam-gpu-lr:0.01,decay:0.0005-layers-compress-rates-${compress_rate}-percent-float32.txt 2>&1 &
         CUDA_PID=$!
         wait ${CUDA_PID}
    done;
done;
