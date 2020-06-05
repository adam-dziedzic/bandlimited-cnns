#!/usr/bin/env bash
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main_adversarial.py --adam_beta2=0.999 --compress_type='STANDARD' --compress_rates 0 --conv_type="STANDARD2D" --conv_exec_type=CUDA --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=1000 --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='PINNED' --min_batch_size=32 --model_path="pretrained" --momentum=0.9 --next_power2='FALSE' --optimizer_type='MOMENTUM' --preserve_energies=100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='FALSE' --weight_decay=0.0005 --workers=4 --precision_type=FP32 --only_train=FALSE --test_compress_rate='FALSE' --noise_sigma=0.0 --noise_sigmas 0.02 --noise_epsilon=0.0 --noise_epsilons 0.0 --start_epsilon=0 --attack_type="RECOVERY" --attack_name="CarliniWagnerL2Attack" --start_epoch=0 --network_type='ResNet50' --dataset="imagenet" --compress_fft_layer=0 --values_per_channel=0 --many_values_per_channel 0 --recover_type="gauss" --step_size=1 --many_recover_iterations 0 --many_attack_iterations 0 --many_noise_iterations 0 --use_foolbox_data='FALSE' --laplace_epsilon=0.0 --laplace_epsilons 0 --many_svd_compress 0 --adv_type='BEFORE' --prediction_type='CLASSIFICATION' --prediction_type='CLASSIFICATION' --attack_strengths 0.004 0.04 0.0004 0.4 4.0 --attack_confidence 1000 --target_class=-1 --binary_search_steps=1 --many_attack_iterations 1000 --use_set 'test_set' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup python main_adversarial.py --attack_name="CarliniWagnerL2Attack" --noiseInit 0.0 --noiseInner 0.0 --model_path="vgg16/vgg16-rse_perturb_0.0_init_noise_0.0_inner_noise_0.0_batch_size_128_compress_rate_0.0_none.pth-test-accuracy-0.9356" >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 36730
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness$ echo ${timestamp}.txt
2020-06-05-01-25-11-001606244.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup python main_adversarial.py --attack_name="CarliniWagnerL2Attack" --noiseInit 0.2 --noiseInner 0.1 --model_path="vgg16/vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1_batch_size_128_compress_rate_0.0_gauss.pth-test-accuracy-0.8829" >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 37200
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness$ echo ${timestamp}.txt
2020-06-05-01-28-48-312342977.txt

