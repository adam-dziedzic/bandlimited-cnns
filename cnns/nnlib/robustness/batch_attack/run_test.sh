#!/usr/bin/env bash
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 0.01 --net_mode '0-0' --attack_iters 300 --batch_size 3584 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
skr-compute1

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 100.0 10.0 1.0 0.1 0.01 0.001 --net_mode '0-0' --attack_iters 1 --batch_size 3584 --limit_batch_number 1 --c 0.01 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
skr-compute1

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 0.04 --net_mode '0-0' --attack_iters 300 --batch_size 2048 --limit_batch_number 0 --c 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
nips 1

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 0.035 --net_mode '0-0' --attack_iters 300 --batch_size 2048 --limit_batch_number 0 --c 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
nips 0


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 0.045 --net_mode '0-0' --attack_iters 300 --batch_size 2048 --limit_batch_number 0 --c 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
wifi 0

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 0.032 --net_mode '0-0' --attack_iters 300 --batch_size 2048 --limit_batch_number 0 --c 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
wifi 1

cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ nvidia-smi
Sun Jan 12 20:35:25 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.40.04    Driver Version: 418.40.04    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:03:00.0 Off |                    0 |
| N/A   32C    P0    37W / 250W |  14869MiB / 16280MiB |     83%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P100-PCIE...  Off  | 00000000:82:00.0 Off |                    0 |
| N/A   31C    P0    37W / 250W |  14869MiB / 16280MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0    157322      C   /home/cc/anaconda3/bin/python3.6           14859MiB |
|    1    157390      C   /home/cc/anaconda3/bin/python3.6           14859MiB |

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.04' --attack_iters 300 --batch_size 3584 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 25960
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-15-04-47-51-999248400.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.1' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
nips

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.05' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
nips

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.01' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
iclr

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.02' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
iclr

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.06' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
iclr

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.07' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
iclr

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.03' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
wifi

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.045' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
wifi

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.0-model-0.01' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
icml2

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.0-model-0.02' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
icml2


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.01-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
icml1
[1] 83466
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-15-14-09-42-155628870.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.02-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 83619
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-15-14-10-33-761795890.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.03-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
iclr

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.04-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.045-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.05-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.06-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
wifi

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.07-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.0-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
nips

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.005-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.01-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
icml2

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.01-model-0.0' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0.001 0.0005 0.0001 0.0 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'perturb-0.01-model-0.0' --attack_iters 200 --batch_size 3584 --limit_batch_number 0 --c 0.02 0.03 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 3584 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.03_inner_noise_0.03.pth-test-accuracy-0.8713' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 25569
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-15-20-18-39-408925781.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1.pth-test-accuracy-0.8772' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 196597
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-02-39-08-722970932.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse_perturb_0.0_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.9343' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.9384' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 392
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-02-40-36-384662829.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.04 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.04_inner_noise_0.03.pth-test-accuracy-0.8606' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[4] 490
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-02-42-34-097082809.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 300 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.03_inner_noise_0.03.pth-test-accuracy-0.8713' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 102517
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-02-44-46-258566667.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.04 --noiseInner 0.04 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.04_inner_noise_0.04.pth-test-accuracy-0.8234' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 102603
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-02-46-20-374603532.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.035 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.035_inner_noise_0.03.pth-test-accuracy-0.8646' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 87584
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-02-49-45-653843765.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.045 --noiseInner 0.04 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.045_inner_noise_0.04.pth-test-accuracy-0.8208' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 87664
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-02-50-39-907923243.txt



timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.05 --noiseInner 0.04 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.05_inner_noise_0.04.pth-test-accuracy-0.8119' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 3684
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-08-18-34-589157529.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.06 --noiseInner 0.05 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.06_inner_noise_0.05.pth-test-accuracy-0.7597' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 3785
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-08-19-52-832398549.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.07 --noiseInner 0.06 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.07_inner_noise_0.06.pth-test-accuracy-0.7101' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 3884
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-08-21-08-683568126.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.03 --noiseInner 0.02 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.03_inner_noise_0.02.pth-test-accuracy-0.9016' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[4] 4034
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-08-26-05-205594538.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv-fc' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-conv-fc_perturb_0.03_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.8609' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
wifi

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv-bn' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-conv-bn_perturb_0.03_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.8586' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 40885
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-09-23-02-496807845.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-fc' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-fc_perturb_0.03_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.9337' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 27748
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-09-25-21-413442686.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-bn' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-bn_perturb_0.03_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.9275' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 27827
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-09-26-26-686090374.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 3584 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv-every-2nd' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-conv-every-2nd_perturb_0.0_init_noise_0.03_inner_noise_0.03.pth-test-accuracy-0.9082' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 2745
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-06-17-25-048108313.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-fc' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-fc_perturb_0.03_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.9337' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv-every-3rd' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-conv-every-3rd_perturb_0.0_init_noise_0.03_inner_noise_0.03.pth-test-accuracy-0.9053' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
icml1 120788
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-12-23-39-708469757.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv-even' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-conv-even_perturb_0.0_init_noise_0.03_inner_noise_0.03.pth-test-accuracy-0.8979' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 120872
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-12-25-42-144823291.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-weight' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-weight_perturb_0.03_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.8508' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 105553
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-12-29-30-500399575.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-fc-bn' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-fc-bn_perturb_0.03_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.9281' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 105660
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-12-32-07-584447813.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-fc-bn' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-fc-bn_perturb_0.03_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.9281' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse-perturb_perturb_0.03_init_noise_0.2_inner_noise_0.1.pth-test-accuracy-0.7752' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 26745
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-16-19-47-50-230360246.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.03_inner_noise_0.03_batch_size_128.pth-test-accuracy-0.8632' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 41072
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-01-56-13-777983965.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.03_inner_noise_0.03_batch_size_64.pth-test-accuracy-0.8599' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 41162
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-01-57-19-647911611.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.03_inner_noise_0.03_batch_size_32.pth-test-accuracy-0.847' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 41274
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-01-59-17-253172208.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1.pth-test-accuracy-0.7851' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[4] 41359
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-02-02-14-703511978.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-conv_perturb_0.0_init_noise_0.03_inner_noise_0.03_batch_size_256.pth-test-accuracy-0.8517' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 60320
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-02-06-30-921146739.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.03 --noiseInner 0.03 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-conv_perturb_0.0_init_noise_0.03_inner_noise_0.03_batch_size_512.pth-test-accuracy-0.8375' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 60412
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-02-07-25-014942128.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb-weights' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-rse-perturb-weights_perturb_0.03_init_noise_0.2_inner_noise_0.1_batch_size_128.pth-test-accuracy-0.7664' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 47254
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-02-10-10-142521190.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.02 --modelIn '../../pytorch_architecture/vgg16/vgg16-rse-perturb_perturb_0.02_init_noise_0.2_inner_noise_0.1_batch_size_128.pth-test-accuracy-0.8178' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 47349
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-02-11-45-107367963.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.04 --noiseInner 0.04 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.04_inner_noise_0.04_batch_size_128.pth-test-accuracy-0.8198' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 128880
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-11-16-42-531404943.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-conv' --noiseInit 0.05 --noiseInner 0.05 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.05_inner_noise_0.05_batch_size_128.pth-test-accuracy-0.7754' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 128965
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-11-17-53-812967049.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.0 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1.pth-test-accuracy-0.7851' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 46444
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-11-23-50-228387291.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb-weights' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-rse-perturb-weights_perturb_0.03_init_noise_0.2_inner_noise_0.1_batch_size_128.pth-test-accuracy-0.7664' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 50515
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-11-30-34-873657162.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.02 --modelIn '../../pytorch_architecture/vgg16/vgg16-rse-perturb_perturb_0.02_init_noise_0.2_inner_noise_0.1_batch_size_128.pth-test-accuracy-0.8178' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 50595
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-11-30-58-531762017.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.01 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1.pth-test-accuracy-0.8772' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 28713
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-05-34-33-261024385.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.02 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1.pth-test-accuracy-0.8772' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 47006
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-12-08-39-905179909.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.015 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1.pth-test-accuracy-0.8772' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 47082
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-12-09-22-559677243.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'rse-perturb' --noiseInit 0.2 --noiseInner 0.1 --paramNoise 0.005 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1.pth-test-accuracy-0.8772' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 47153
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-12-09-53-984737977.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.3 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.3_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.7837' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 83284
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-01-47-29-510058099.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -1.0 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-rse_perturb_-1.0_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.9356' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 3683
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-17-22-10-59-783838216.txt



timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.5 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.5_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.8379' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 174006
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-13-15-315714133.txt



timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.03 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-rse_perturb_-0.03_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.7082' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 97631
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-15-23-375431201.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.1 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-rse_perturb_-0.03_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.7082' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt



timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.2 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-rse_perturb_-0.01_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.7009' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 1393
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-19-22-730413941.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.3 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-rse_perturb_-0.01_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.7009' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 1496
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-19-57-770855144.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.8 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-rse_perturb_-1.0_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.9356' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 70281
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-27-20-672251151.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.4 --modelIn '../../pytorch_architecture/vgg16/vgg16-perturb-rse_perturb_-1.0_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.9356' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 70389
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-27-39-188482040.txt



timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.6 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.6_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.8669' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 108045
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-31-11-655947040.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.7 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.7_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.8902' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 108207
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-32-59-950636795.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.8 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.8_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.9094' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 108310
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-04-33-56-143697365.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.9 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.9_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.928' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 178392
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-12-59-33-257285831.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise 0.1 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_0.1_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.6681' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 178517
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-13-04-53-949785487.txt



timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.1 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.1_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.7247' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 2155
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-07-06-52-760829959.txt



timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.2 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.2_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.7629' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 112859
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-13-08-33-052337834.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.3 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.3_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.7837' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 112990
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-13-09-21-911863291.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'custom' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 --defense 'perturb-rse' --noiseInit 0.0 --noiseInner 0.0 --paramNoise -0.4 --modelIn '../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-rse_perturb_-0.4_init_noise_0.0_inner_noise_0.0_batch_size_128.pth-test-accuracy-0.8111' >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 113090
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-13-10-07-737860682.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '3-0' --defense 'rse' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 180842
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-20-25-56-483580383.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '3-0' --defense 'rse' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'fft' --noise_epsilons 0.0 --net_mode '0-0' --defense 'rse' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 180960
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-01-18-20-27-51-590980709.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'fft' --noise_epsilons 0.0 --net_mode '0-0' --defense 'rse' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'fft_adaptive' --noise_epsilons 0.0 --net_mode '0-0' --defense 'rse' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
k1
81390



timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '2-1-non-adaptive' --defense 'rse-non-adaptive' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 3414
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-04-09-27-23-794036308.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '2-1' --defense 'rse' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 81035
cc@k-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-04-15-29-06-693902924.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '0-0' --defense 'rse' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 81688
cc@k-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-04-15-37-47-978796348.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'fft_adaptive' --noise_epsilons 50.0 --net_mode '0-0' --defense 'rse' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 85236
cc@z:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-04-17-52-20-399063159.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'fft' --noise_epsilons 50.0 --net_mode '0-0' --defense 'rse' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '2-1-non-adaptive' --defense 'rse-non-adaptive' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 200 400 800 1000 1500 2000 3000 4000 8000 10000 20000 40000  >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 14402
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-04-12-02-37-720004312.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '2-1-non-adaptive' --defense 'rse-non-adaptive' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 80000 160000 320000 640000 1000000 2000000 4000000 8000000 10000000 100000000 100000000 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 70514
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-04-22-12-46-328963895.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '2-1' --defense 'rse' --attack_iters 200 --batch_size 2048 --limit_batch_number 0 --c 100.0 10.0 2.0 1.0 0.5 0.4 0.3 0.2 0.1 0.07 0.05 0.04 0.03 0.02 0.01 0.005 0.001 0.0005 0.0001 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 71068
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-04-23-03-35-874460844.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '2-0' --defense 'rse' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 72305
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-05-01-11-21-813337529.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '3-0' --defense 'rse' --attack_iters 200 --batch_size 1024 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 72307
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-05-01-11-21-841299955.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80' --defense 'fft' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
|    0     33272      C   /home/cc/anaconda3/bin/python3.6           14431MiB |
+-----------------------------------------------------------------------------+
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$
cat 2020-02-06-16-26-57-766001181.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80-non-adaptive' --defense 'fft' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 33399
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-06-16-28-16-360727627.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80-non-adaptive' --defense 'fft' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 34603
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-06-18-17-13-652892443.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80' --defense 'fft' --attack_iters 1 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
2020-02-06-18-06-31-557503601.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80-non-adaptive' --defense 'fft' --attack_iters 10 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 34735
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-06-18-19-51-922603122.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80' --defense 'fft' --attack_iters 10 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 166003
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-06-18-42-07-797495622.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80-non-adaptive' --defense 'fft' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 38814
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-16-03-20-45-947840863.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80' --defense 'fft' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 38579
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-16-03-15-13-966271205.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'vgg-fft-80' --defense 'fft' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --compress_rate 80.0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 24367
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-02-15-21-19-58-487822186.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '0-1' --defense 'rse' --noise_form 'gauss' --attack_iters 200 --batch_size 16 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '0-1' --defense 'rse' --noise_form 'gauss' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode '1-1' --defense 'rse' --noise_form 'gauss' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
2020-06-03-14-17-27-597260691.txt

vgg16-rse_perturb_0.0_init_noise_0.15_inner_noise_0.1_batch_size_128_compress_rate_0.0_laplace.pth-test-accuracy-0.8809
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'generic' --defense 'rse' --noise_form 'laplace' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --noiseInit 0.15 --noiseInner 0.1 --net 'vgg16' --modelIn 'vgg16/vgg16-rse_perturb_0.0_init_noise_0.15_inner_noise_0.1_batch_size_128_compress_rate_0.0_laplace.pth-test-accuracy-0.8809' --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 32461
(abs) ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-06-04-14-52-30-452031824.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'generic' --defense 'rse' --noise_form 'laplace' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --noiseInit 0.15 --noiseInner 0.1 --net 'vgg16' --modelIn 'vgg16/vgg16-rse_perturb_0.0_init_noise_0.15_inner_noise_0.1_batch_size_128_compress_rate_0.0_laplace.pth-test-accuracy-0.8809' --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 3958
(abs) ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-06-04-14-53-56-165223945.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'generic' --defense 'rse' --noise_form 'uniform' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16' --modelIn 'vgg16/vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1_batch_size_128_compress_rate_0.0_uniform.pth-test-accuracy-0.8782' --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 57115
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-06-04-20-03-44-387903653.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'generic' --defense 'rse' --noise_form 'gauss' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16' --modelIn 'vgg16/vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1_batch_size_128_compress_rate_0.0_gauss.pth-test-accuracy-0.8829' --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 980
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-06-04-20-07-27-845311806.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'generic' --defense 'rse' --noise_form 'laplace' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --noiseInit 0.15 --noiseInner 0.1 --net 'vgg16' --modelIn 'vgg16/vgg16-rse_perturb_0.0_init_noise_0.15_inner_noise_0.1_batch_size_128_compress_rate_0.0_laplace.pth-test-accuracy-0.8809' --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 1422
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-06-04-20-10-11-823417433.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'generic' --defense 'rse' --noise_form 'laplace' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --noiseInit 0.15 --noiseInner 0.1 --net 'vgg16' --modelIn 'vgg16/vgg16-rse_perturb_0.0_init_noise_0.15_inner_noise_0.1_batch_size_128_compress_rate_0.0_laplace.pth-test-accuracy-0.8592' --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 45937
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-06-05-11-41-43-718685872.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'generic' --defense 'rse' --noise_form 'uniform' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --noiseInit 0.3 --noiseInner 0.2 --net 'vgg16' --modelIn 'vgg16/vgg16-rse_perturb_0.0_init_noise_0.3_inner_noise_0.2_batch_size_128_compress_rate_0.0_uniform.pth-test-accuracy-0.8679' --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 45630
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/batch_attack$ echo ${timestamp}.txt
2020-06-05-11-39-42-065564657.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup python attack.py --channel 'empty' --noise_epsilons 0.0 --net_mode 'generic' --defense 'rse' --noise_form 'gauss' --attack_iters 200 --batch_size 512 --limit_batch_number 0 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16' --modelIn 'vgg16/vgg16-rse_perturb_0.0_init_noise_0.2_inner_noise_0.1_batch_size_128_compress_rate_0.0_gauss.pth-test-accuracy-0.8829' --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt