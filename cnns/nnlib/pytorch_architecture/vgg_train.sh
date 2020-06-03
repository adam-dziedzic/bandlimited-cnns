#!/usr/bin/env bash
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
[1] 24121
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-03-13-02-805967604.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-conv-bn'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 24237
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-03-13-47-062298473.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv-every-2nd'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 13575
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-02-54-36-854774647.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv-even'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 104597
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-08-59-27-757019512.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv-every-3rd'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 105625
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-09-08-41-262621732.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-fc-bn'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 89976
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-09-03-51-444986162.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-weight'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 90081
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-09-04-16-872948080.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16-rse-perturb'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 20449
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-11-34-31-130518716.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 128 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 9421
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-17-59-14-528687601.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 64 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 9530
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-17-59-37-391030220.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 32 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 9663
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-00-14-687866631.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 16 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[4] 9796
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-00-48-245321187.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 8 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 122656
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-01-56-937652977.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 4 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 122814
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-03-05-288923895.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 2 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 107272
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-03-54-054027596.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 1 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 107362
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-04-27-892680312.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 256 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 43480
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-05-44-035685847.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 512 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 43593
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-06-22-059926369.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.02 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16-rse-perturb'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 30451
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-09-45-832886385.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.03 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16-rse-perturb-weights'>> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 30545
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-16-18-10-02-963129804.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 16 --lr 0.01 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 130997
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-17-02-15-35-932217058.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.03 --noiseInner 0.03 --net 'vgg16-perturb-conv' --batchSize 8 --lr 0.01 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 130999
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-17-02-15-35-954453444.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.04 --noiseInner 0.04 --net 'vgg16-perturb-conv' --batchSize 128 --lr 0.1 --initializeNoise 0.03 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 111577
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-17-02-29-22-166827512.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.05 --noiseInner 0.05 --net 'vgg16-perturb-conv' --batchSize 128 --lr 0.1 --initializeNoise 0.04 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 111703
cc@icml2:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-17-02-30-08-663760220.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.04 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16-rse-perturb' --initializeNoise 0.03 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 63883
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-17-12-12-30-076554215.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.05 --noiseInit 0.2 --noiseInner 0.1 --net 'vgg16-rse-perturb' --initializeNoise 0.04 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 63974
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-17-12-12-54-119473010.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.1 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 27142
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-17-13-38-36-006390363.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.2 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.3 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.4 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.1 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.5 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.01 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.05 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.02 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.03 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.15 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.04 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -1.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 21701
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-17-19-34-37-549409155.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.6 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[2] 83531
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-18-01-59-46-899250464.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.7 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 83533
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-18-01-59-46-920549879.txt


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.8 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[4] 84562
cc@iclr:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-18-02-04-36-849954604.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise -0.9 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-perturb-rse' --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 167262
cc@icml1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-01-18-02-06-34-215712018.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-fft' --compress_rate 85.0 --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 4692
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-02-05-14-49-53-716081202.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.1 --net 'vgg16-rse' --compress_rate 0.0 --initializeNoise 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup python vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.1 --net 'vgg16-rse' --compress_rate 0.0 --initializeNoise 0.0 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 2026
(abs) ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-06-02-17-41-27-156714207.txt

