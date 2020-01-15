timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_perturb_train.py --param_noise 0.04 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt