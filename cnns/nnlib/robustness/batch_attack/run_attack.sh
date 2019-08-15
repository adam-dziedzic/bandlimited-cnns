#!/usr/bin/env bash

c="0.0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0,10.0,100.0"

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
# model_in=./${net}/brelu_0.05.pth
# model_in=./${net}/rse_0.1_0.0_ady.pth-test-accuracy-0.8504
model_in=./${net}/rse_0.0_0.0_ady.pth-test-accuracy-0.8523
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
#c=0.01
noise_init=0.0
noise_inner=0.0
mode=test
ensemble=1
channel='empty'
noise_epsilon=0.03
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_${channel}_ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 attack.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} --channel ${channel} --noise_epsilon=${noise_epsilon} > ${log} 2>&1 &
echo ${log}


#!/usr/bin/env bash

c="0.0 0.0001 0.0005 0.001 0.005 0.01 0.03 0.04 0.05 0.1 0.5 1.0 2.0 10.0"
device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
# model_in=./${net}/brelu_0.05.pth
# model_in=./${net}/rse_0.1_0.0_ady.pth-test-accuracy-0.8504
model_in=./${net}/rse_0.0_0.0_ady.pth-test-accuracy-0.8523
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
#c=0.01
noise_init=0.0  # the noise added in the initial random noise layer in the network
noise_inner=0.0  # the noise added in the internal random noise layers in the network
mode=test
ensemble=1
channel='fft'
noise_epsilon=50
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_${channel}_ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 attack.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} --channel ${channel} --noise_epsilon=${noise_epsilon} >> ${log} 2>&1 &
echo ${log}

