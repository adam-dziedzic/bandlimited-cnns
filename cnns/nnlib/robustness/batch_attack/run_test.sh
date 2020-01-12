timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 0.01 --net_mode '0-0' --attack_iters 300 --batch_size 3584 --limit_batch_number 0 --c 0.0 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.07 0.1 0.2 0.3 0.4 0.5 1.0 2.0 10.0 100.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
skr-compute1

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../ nohup /home/${USER}/anaconda3/bin/python3.6 attack.py --channel 'perturb' --noise_epsilons 100.0 10.0 1.0 0.1 0.01 0.001 --net_mode '0-0' --attack_iters 1 --batch_size 3584 --limit_batch_number 1 --c 0.01 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
skr-compute1