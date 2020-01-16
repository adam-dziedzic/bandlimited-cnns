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
