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

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16-rse'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 133394
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-18-38-26-296271210.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 133771
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-15-18-42-21-102079378.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-rse'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.1 --noiseInner 0.1 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.02 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

iclr

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.04 --noiseInner 0.03 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.05 --noiseInner 0.04 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.06 --noiseInner 0.05 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.07 --noiseInner 0.06 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

cc@129.114.108.13 icml1
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.04 --noiseInner 0.04 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

ssh cc@129.114.108.29 icml2
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.035 --noiseInner 0.03 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.045 --noiseInner 0.04 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

ssh cc@129.114.108.45 wifi
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.045 --noiseInner 0.03 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.045 --noiseInner 0.035 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

ssh cc@129.114.108.143 nips
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.05 --noiseInner 0.05 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.06 --noiseInner 0.06 --net 'vgg16-perturb-conv'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-fc'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 11233
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-03-10-05-269879544.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-bn'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 11341
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-03-10-43-772897143.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-conv-fc'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt