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


