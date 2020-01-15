timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.04 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 173993
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-03-37-38-732530576.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.045 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 174179
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-03-39-30-244579043.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.05 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 112604
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-03-41-01-603297586.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.1 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 112717
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-03-41-46-619681467.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.01 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 79022
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-03-45-12-980256054.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 79186
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-03-46-12-704017833.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.06 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 79308
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-03-46-40-388416319.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.07 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[4] 79441
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-03-47-12-818883295.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.04 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.035 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.005 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt