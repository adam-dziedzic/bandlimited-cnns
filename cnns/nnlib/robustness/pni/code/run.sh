#!/usr/bin/env bash

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --adv_eval --epoch_delay 5 \
    --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
g1

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_input
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --adv_eval --epoch_delay 5 \
    --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
g2

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_both
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --adv_eval --epoch_delay 5 \
    --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

[1] 169018
cc@sat:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-04-14-04-57-058607111.txt


# no adversarial training

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_input
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

[1] 71214
cc@icml-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-04-14-11-03-294585424.txt

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_both
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

[1] 119881
cc@icml-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-04-14-19-45-778648234.txt

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

[1] 113589
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-04-14-21-24-566295267.txt

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_input
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --adv_eval --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 74651
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-02-53-25-778977624.txt

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_both
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --adv_eval --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 66643
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-03-00-16-357489616.txt


############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_weight
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

[1] 75452
cc@sat:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-03-21-57-962498895.txt


############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_both
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 73745
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-03-25-28-247734754.txt

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_input
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 121227
cc@z:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-03-27-16-250630896.txt

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_both
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

[1] 102368
cc@wifi:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-04-08-31-621040073.txt

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 58848
cc@sat:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-05-37-23-891561239.txt



############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 6965
cc@z:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-05-36-01-105993017.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 13099
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-10-33-49-546921320.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_both
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net_both
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \m
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 100998
-end 29
cc@icml-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-16-40-40-277973375.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_vanilla_resnet20
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \m
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 71337
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-16-51-32-572125767.txt

cc@f:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$
[1]   Done                    PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} --data_path ${data_path} --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} --epochs ${epochs} --learning_rate 0.1 --optimizer ${optimizer} --schedule 80 120 --gammas 0.1 0.1 --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 --print_freq 100 --decay 0.0003 --momentum 0.9 --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1  (wd: ~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code)
(wd now: ~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture)


==>>[2020-02-05 21:58:12] [Epoch=159/160] [Need: 00:01:55] [LR=0.0010][M=0.90] [Best : Accuracy=84.15, Error=15.85]                           Epoch: [159][000/391]   Time 0.492 (0.492)   Data 0.199 (0.199)   Loss 0.7769 (0.7769)   Prec@1 93.750 (93.750)   Prec@5 99.219 (99.219)   [2020-02-05 21:58:12]                                                                                                                        Epoch: [159][100/391]   Time 0.304 (0.312)   Data 0.000 (0.002)   Loss 0.7494 (0.7410)   Prec@1 88.281 (88.877)   Prec@5 100.000 (99.613)   [2020-02-05 21:58:43]
  Epoch: [159][200/391]   Time 0.344 (0.300)   Data 0.000 (0.001)   Loss 0.8205 (0.7450)   Prec@1 87.500 (89.043)   Prec@5 98.438 (99.677)   [2020-02-05 21:59:12]
  Epoch: [159][300/391]   Time 0.327 (0.299)   Data 0.000 (0.001)   Loss 0.6536 (0.7461)   Prec@1 80.469 (88.741)   Prec@5 100.000 (99.676)   [2020-02-05 21:59:42]                                                                                                                       **Train** Prec@1 88.730 Prec@5 99.684 Error@1 11.270
  **Adversarial Train** Prec@1 53.920 Prec@5 97.768 Error@1 46.080                                                                            **Test** Prec@1 83.760 Prec@5 99.270 Error@1 16.240
---- save figure the accuracy/loss curve of train/val into ./save//cifar10_vanilla_resnet20_160_SGD_train_layerwise_3e-4decay_robust_net/curve.png


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_input
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net_input
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 130484
cc@icml-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-18-52-11-960855088.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_robust_net
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 2 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 70468
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-19-41-09-712288797.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust # init 0.1 inner 0.1
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_robust_net_adv_train
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 79546
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-20-00-18-937669813.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-fft' --compress_rate 85.0 --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[1] 4844
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-02-05-14-52-24-658630568.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-fft' --compress_rate 80.0 --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
f 100995
2020-02-05-21-14-46-235774981.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-fft' --compress_rate 70.0 --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[3] 101291
cc@f:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-02-05-21-15-47-337316762.txt

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 vgg_train.py --paramNoise 0.0 --noiseInit 0.0 --noiseInner 0.0 --net 'vgg16-fft' --compress_rate 70.0 --initializeNoise 0.02 >> ${timestamp}.txt 2>&1 &
echo ${timestamp}.txt
[4] 101366
cc@f:~/code/bandlimited-cnns/cnns/nnlib/pytorch_architecture$ echo ${timestamp}.txt
2020-02-05-21-16-27-074140847.txt



PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_robust_net_init_noise_0.15
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 77816
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-21-26-54-710179415.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_vanilla_resnet20_plain
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 56312
cc@icml-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-06-05-30-25-346400537.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_vanilla_resnet20_plain_no_adv
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 111266
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-06-07-47-05-931286438.txt
[1] 23458
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-06-16-33-52-502708202.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_vanilla_resnet20_plain
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 112258
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-06-07-49-34-905649455.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 62743
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-06-07-53-56-198637208.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 62419
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-06-07-53-08-232294426.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 120954
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-06-07-56-07-651468314.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_input
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 121461
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-06-07-58-23-439637304.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets/tiny-224/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 137953
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-03-38-56-802012326.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets/tiny-224/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
g2
[2] 138292
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-03-39-59-518614900.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=imagenet
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets/tiny-224/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02
dataset=imagenet
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets/tiny-224/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=imagenet
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets/tiny-224/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=imagenet
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets/tiny-224/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=imagenet
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets/tiny-64/"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 90458
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-10-11-635161494.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 90645
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-14-39-557431445.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 90825
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-15-13-385413955.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 91994                                                                                                                                   cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-16-37-444371274.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 92817
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-17-21-435364545.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters 40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 78569
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-27-54-835694871.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters=40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 79151
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-29-13-558159540.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters 40 \
   >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 80315
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-30-08-183339905.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 80658
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-30-27-956193475.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters 40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 81609
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-36-24-783102383.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters 40 \
   >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 6244
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-08-23-32-29-696181815.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters=40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 82117
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-37-38-391999029.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 48591
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-39-46-508918889.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters=40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 48916
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-40-19-667773368.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=stl10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters=40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 52832
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-43-24-525493119.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=stl10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters=40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 52920
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-44-27-931238848.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters 40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 78569
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-27-54-835694871.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-100-iters
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters=100 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 53312
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-49-47-737908358.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-100-iters
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters=100 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 54479
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-09-05-52-24-203653053.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-attack-iters-40
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 123446
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-17-24-21-783253335.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-attack-iters-7
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 7 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 123591
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-17-24-38-313852566.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_013
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-attack-iters-7
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 7 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 34326
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-19-50-26-595321624.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_013
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-attack-iters-7
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 40 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 171687
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-19-51-12-835446430.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_013
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-attack-iters-7-adv-train
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 7 --adv_train \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 172810
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-19-53-04-872641562.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_013
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-attack-iters-7-adv-train
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 40 --adv_train \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 173360
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-19-53-40-329671271.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_014
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-roubst-014-no-adv-train
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 0 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 63686
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-21-09-46-773395157.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_014
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-roubst-014-no-adv-train
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 40 \
    --adv_train \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 20073
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-21-20-08-929333746.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_014
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-roubst-014-no-adv-train
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 100 \
    --adv_train \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 20619
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-21-20-44-493891179.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_robust_0-9_0-7
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.09 \
    --inner_noise 0.07 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 31204
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-16-11-56-427824622.txt

[1] 50935
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-22-13-32-685014413.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_robust_0-08_0-08
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08 \
    --inner_noise 0.08 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 52296
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-22-16-04-193327018.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_robust_0-1_0-09
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.1 \
    --inner_noise 0.09 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 53931
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-22-18-15-051988181.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_robust_0-1_0-09
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08 \
    --inner_noise 0.07 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 54476
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-22-18-35-303368930.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_014
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-roubst-014-no-adv-train
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 40 \
    --adv_train \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 183636
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-11-04-25-44-048463992.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_014
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-roubst-014-no-adv-train-true
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 0 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 184099
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-11-04-26-39-189445535.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_robust_0-9_0-7
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 185585
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-11-04-28-01-641968720.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_only
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 186864
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-11-04-29-13-520212968.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_013
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-roubst-013-adv-train
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 40 \
    --adv_train \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 96554
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-11-04-31-58-108970222.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_013
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-roubst-013-no-adv-train-true
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --attack_iters 0 \
    >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 96706
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-11-04-32-19-562397893.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_robust_0-1_0-09
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.1 \
    --inner_noise 0.09 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 71933
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-11-04-34-17-796533109.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar100
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_robust_0-1_0-09
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.1 \
    --inner_noise 0.09 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 22472
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-10-22-37-30-279744125.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_robust-0.07-0.06
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.07 \
    --inner_noise 0.06 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 114431
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-12-02-14-58-101825118.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_robust-0.07-0.06
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.07 \
    --inner_noise 0.06 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 114820
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-12-02-15-51-126110622.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_robust-0.05-0.04
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.05 \
    --inner_noise 0.04 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 125712
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-12-02-38-49-734586301.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_robust-0.05-0.04
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.05 \
    --inner_noise 0.04 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 125856
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-12-02-39-14-445383221.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_only
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_iters 100 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 88151
cc@i-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-25-15-57-04-273719851.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters 40 \
   >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 88512
cc@i-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-25-15-57-39-228843915.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_robust_0-14_0-10
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.14 \
    --inner_noise 0.10 --adv_train \
    --attack_iters 100 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[5] 90499
cc@i-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-25-16-01-22-359838561.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-without-adv-train_plain_robust_0-14_0-10
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.14 \
    --inner_noise 0.10 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[6] 91406
cc@i-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-25-16-03-10-170347657.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-no-adv-train_only
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 86138
cc@i-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-25-17-09-52-069740438.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters 40 \
   >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 86237
cc@i-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-25-17-10-11-386067412.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_robust-0.07-0.06
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.07 \
    --inner_noise 0.06 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 87471
cc@i-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-25-17-11-51-338507251.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-without-adv-train_robust-0.07-0.06
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.07 \
    --inner_noise 0.06 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 88092
cc@i-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-25-17-12-25-568371784.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-without-adv-train_robust-0.05-0.04
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.05 \
    --inner_noise 0.04 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 27527
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-26-21-31-01-483688075.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-without-adv-train_robust-0.04-0.03
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.04 \
    --inner_noise 0.03 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 27661
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-26-21-31-58-096840788.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-without-adv-train_robust-0.03-0.02
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.03 \
    --inner_noise 0.02 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 27777
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-26-21-32-38-278049122.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_robust_0-14_0-10
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.14 \
    --inner_noise 0.10 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 8884
cc@rtx:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-29-18-53-34-770940567.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train --attack_iters 40 \
   >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-without-adv-train_robust-0.04-0.03
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.04 \
    --inner_noise 0.03 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 9462
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-01-21-31-27-485749832.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-without-adv-train_robust-0.04-0.03
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.04 \
    --inner_noise 0.04 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 158873
cc@i-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-32-16-704481602.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-without-adv-train_robust-0.03-0.03
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.03 \
    --inner_noise 0.03 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 159247
cc@i-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-33-06-302830089.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_robust-0.07-0.06
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.07 \
    --inner_noise 0.06 --adv_train \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 160607
cc@i-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-34-55-754752844.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay-adv-train_robust-0.08-0.07
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08 \
    --inner_noise 0.07 --adv_train \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 161556
cc@i-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-35-47-804009354.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=adv-train-0.09-0.08
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.09 \
    --inner_noise 0.08 --adv_train \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 2952
cc@rtx:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-37-37-294899008.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=adv-train-0.09-0.08
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.09 \
    --inner_noise 0.08 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 192870
cc@i-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-39-10-341006193.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=adv-train-0.08-0.07
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08 \
    --inner_noise 0.07 --adv_train \
    --attack_iters 40 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 193327
cc@i-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-40-29-748117820.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train-0.25-0.21
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.25 \
    --inner_noise 0.21 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 195243
cc@i-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-43-45-118528090.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train-0.3-0.2
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.3 \
    --inner_noise 0.2 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[4] 195598
cc@i-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-02-03-44-38-792221701.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=adv-train-0.08-0.07
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08\
    --inner_noise 0.07 --adv_train \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 17097
cc@rtx:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-03-17-52-43-686410941.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.03-0.02
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.03 \
    --inner_noise 0.02 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 22511
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-03-11-59-03-207700069.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.02-0.01
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.02 \
    --inner_noise 0.01 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 14028
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-04-11-23-31-285120306.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=adv-train-0.07-0.06
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.07 \
    --inner_noise 0.06 --adv_train \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 34523
cc@rtx:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-04-17-28-15-091439370.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.08-0.07
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08 \
    --inner_noise 0.07 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 10730
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-04-14-42-18-135400803.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.09-0.08
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.09 \
    --inner_noise 0.08 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 18056
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-05-19-53-35-772051178.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.09-0.08
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.09 \
    --inner_noise 0.08 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 99000
cc@m-3:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-06-03-20-54-406689261.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.1-0.09
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.1 \
    --inner_noise 0.09 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 101138
cc@m-3:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-06-03-29-25-885235788.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_strengths 0.031 \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 19729
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-18-22-11-23-453104281.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_strengths 0.031 \
    --attack_iters 7 \
   >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 13460
cc@rtx:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-19-03-12-50-034552005.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.08-0.07
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08 \
    --inner_noise 0.07 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 19729
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-18-22-11-23-453104281.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.08-0.07
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08 \
    --inner_noise 0.07 \
    --attack_strengths 0.0 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 159439
cc@i:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-19-04-18-38-012564140.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=adv-train_robust-0.08-0.07
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.08 \
    --inner_noise 0.07 \
    --attack_strengths 0.031 \
    --adv_train \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 160024
cc@i:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-19-04-20-12-960935239.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_strengths 0.031 \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[3] 161680
cc@i:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-19-04-22-24-078882321.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_strengths 0.031 \
    --attack_iters 7 \
   >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.09-0.08
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.09 \
    --inner_noise 0.08 \
    --attack_strengths 0.0 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 6776
cc@rtx:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-19-16-04-22-388666599.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.1-0.09
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --init_noise 0.1 \
    --inner_noise 0.09 \
    --attack_strengths 0.0 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 31104
cc@rtx:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-19-22-42-15-543433057.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no-adv-train_robust-0.2-0.1
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 \
    --init_noise 0.2 \
    --inner_noise 0.1 \
    --attack_strengths 0.0 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 43363
cc@rtx:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-03-20-01-24-37-190837110.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=no_adv_train_vanilla_pure
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 \
    --init_noise 0.0 \
    --inner_noise 0.0 \
    --attack_strengths 0.0 \
    --attack_iters 0 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[1] 94857
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-04-01-19-24-05-380748062.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=svhn
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay
#dataset path
data_path="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/datasets"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py \
    --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} \
    --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --epoch_delay 5 --adv_train \
    --attack_strengths 0.031 \
    --attack_iters 7 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt