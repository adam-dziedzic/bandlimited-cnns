#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
TENSORBOARD="/home/${USER}/anaconda3/bin/tensorboard"

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir -p ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_input
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=eval_layerwise_resnet20


data_path="/home/${USER}/data/pytorch/cifar10" #dataset path
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info}/tb_log  #tensorboard log path

# set the pretrained model path
# pretrained_model=/home/elliot/Documents/CVPR_2019/CVPR_2019_PNI/code/save/cifar10_noise_resnet20_160_SGD_29_PNI-W/model_best.pth.tar
# pretrained_model=/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/2020-02-03/cifar10_noise_resnet20_160_SGD_train_channelwise_3e-4decay/model_best.pth.tar
pretrained_model=/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/save_adv_train_cifar10_noise_resnet20_input_160_SGD_train_layerwise_3e-4decay/mode_best.pth.tar

############### Neural network ############################
{
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --evaluate --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> eval_${timestamp}.txt 2>&1 &
echo eval_${timestamp}.txt
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait


############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_both
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_noise_both_scaled_evaluate
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_both_160_SGD_train_layerwise_3e-4decay/model_best.pth.tar"
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
    --evaluate --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo train_${timestamp}.txt
[2] 32908
cc@sat:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-03-07-54-093302028.txt


############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_input
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_noise_input_scaled_evaluate
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_input_160_SGD_train_layerwise_3e-4decay/model_best.pth.tar"
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
    --evaluate --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 90255
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo train_${timestamp}.txt
train_2020-02-05-03-09-58-826649672.txt



############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_noise_weight_scaled_evaluate
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/save_adv_train_cifar10_noise_resnet20_weight_160_SGD_train_layerwise_3e-4decay/model_best.pth.tar"
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
    --attack_carlini_eval --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
skr-compute1
test_2020-02-04-23-15-05-106074704.txt



############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_both
dataset=cifar10
epochs=160
batch_size=128
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_noise_both
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save//model_best.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 4 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --adv_eval --evaluate --epoch_delay 5 >> train_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt

############### Configurations ########################
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_both
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_noise_both_eval_carlini
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save//model_best.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 8 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_carlini_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 62935
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-05-05-33-13-724803833.txt



PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_both
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_noise_both_eval_carlini
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_both_160_SGD_train_layerwise_3e-4decay_no_adv_train/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_carlini_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 97261
cc@icml-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-05-05-41-56-513704895.txt



PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net_eval_carlini
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_carlini_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt

cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net
PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net_eval_carlini_0.1
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_carlini_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 99447
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-05-21-11-12-389189283.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_vanilla_net_eval_carlini
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_train_layerwise_3e-4decay_robust_net/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_carlini_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[5] 107060
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-05-22-19-27-502676125.txt

cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net_init_noise_0.15

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net_0.15-0.1_eval_carlini
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net_init_noise_0.15/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_carlini_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 106904
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-05-23-39-33-549076258.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_noise_weight_scaled_evaluate_pgd
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/save_adv_train_cifar10_noise_resnet20_weight_160_SGD_train_layerwise_3e-4decay/model_best.pth.tar"
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
    --attack_eval --attack 'pgd' --attack_iters 10 --resume ${pretrained_model} \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 24576
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-05-19-14-21-025639460.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_no_adv_train_robust_net_eval_carlini
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/save_cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' \
    --attack_iters 10 --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt



PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_robust_net_0.1-0.1_eval_carlini
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net_adv_train/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_carlini_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 48857
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-03-18-42-135144659.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net_init_noise_0.15/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 109538
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-03-51-52-701853014.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_train_robust_net_0.1-0.1_eval_carlini
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_160_SGD_train_layerwise_3e-4decay_robust_net_adv_train/model_best.pth.tar"
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
    --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 115159
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-04-36-38-513433420.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/pni.pth.tar"
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
    --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 115859
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-04-55-08-571345127.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=cifar10
epochs=160
batch_size=1024
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_0.2.pth.tar"
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
    --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 103401
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-04-57-15-627825880.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar10
epochs=160
batch_size=1024
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_01_adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --adv_eval --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_01_adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 70334
cc@sat:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-05-37-20-303029304.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 116933
cc@f:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-04-51-51-600222099.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/pni.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 117405
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-05-39-44-382709938.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_0.2.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 108118
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-05-41-21-419837071.txt

[1] 14846
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-01-07-20-057884541.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_0.2.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 15112
ady@skr-compute1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-06-01-13-23-661644334.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_pni_weight.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 78872
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-02-53-03-038960053.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_pni_input.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 79053
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-02-54-07-126881348.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_pni_weight.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[3] 79326
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-02-56-31-610174452.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_robust_net_01_adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[4] 80662
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-02-58-52-038016137.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_robust_net_01_adv_train.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[5] 81038
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-02-59-30-887853098.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_adv_train.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[6] 83187
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-03-09-45-641006470.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 77999
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-03-12-48-873928028.txt


cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ enable_tb_display=false # enable tensorboard display
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ model=noise_resnet20_robust_02 # + adv. training
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ dataset=cifar10
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ epochs=160
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ batch_size=2560
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ optimizer=SGD
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ # add more labels as additional info into the saving path
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ label_info=train_layerwise_3e-4decay_adv_eval
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_robust_net_02.pth.tar"
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ #dataset path
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ data_path="/home/${USER}/data/pytorch/cifar10"
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
>     --data_path ${data_path}   \
>     --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
>     --epochs ${epochs} --learning_rate 0.1 \
>     --optimizer ${optimizer} \
> --schedule 80 120  --gammas 0.1 0.1 \
>     --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
>     --print_freq 100 --decay 0.0003 --momentum 0.9 \
>     --resume ${pretrained_model} \
>     --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
[6] 83187
cc@nips:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-03-09-45-641006470.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_plain.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 78143
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-09-03-13-42-059583743.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_adv_train_100_iters_pgd_accuracy_83-690.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 116313
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-12-29-075156811.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10-robust-net-adv-train-100-iters-pgd-accuracy-83-860.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 116556
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-18-21-697977954.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_01_pgd_40_iters_accuracy_84-250.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 22967
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-26-49-962378485.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_pgd_attack_40_iters_acc_83.960.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 23137
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-28-38-365062586.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_02_robustnet_acc_80-380.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[3] 23299
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-29-38-294959382.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_weight_pgd_40_iters_attack_accuracy_83-910.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[4] 23533
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-31-36-515585056.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar100
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar100_noise_resnet20_robust_01_160_SGD_train_layerwise_3e-4decay.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar100
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar100_vanilla_resnet20_160_SGD_train_layerwise_3e-4decay.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 144497
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-48-49-464479396.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_02_robustnet_acc_80-380.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[3] 23299
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-29-38-294959382.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_weight_pgd_40_iters_attack_accuracy_83-910.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_pni_weight.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 144821
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-53-11-251398594.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_robust_net_01_adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 145100
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-54-32-156126199.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 111442
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-56-15-080954763.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_robust_net_02.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 111596
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-04-57-20-735208098.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_pni_weight.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' \
    --attack_iters 7 --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 3197
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-13-15-35-016814491.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_robust_net_01_adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 7 --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 3360
cc@g-1:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-13-15-52-561784435.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_adv_train.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' \
    --attack_iters 7 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 116809
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-13-17-42-682609980.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=svhn
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_robust_net_02.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 7 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 116958
cc@g-2:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-13-17-59-636304449.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_02_robustnet_acc_80-380.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 40 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_weight_pgd_40_iters_attack_accuracy_83-910.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 40 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[2] 33912
cc@icml:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-13-22-27-731072464.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_01_pgd_40_iters_accuracy_84-250.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 40 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_pgd_attack_40_iters_acc_83.960.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 40 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_02 # + adv. training
dataset=cifar10
epochs=160
batch_size=2500
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_02_robustnet_acc_80-380.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 40 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_weight_pgd_40_iters_attack_accuracy_83-910.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 40 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_robust_01 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_noise_resnet20_robust_01_pgd_40_iters_accuracy_84-250.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 40 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt


PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=vanilla_resnet20 # + adv. training
dataset=cifar10
epochs=160
batch_size=2560
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_pgd_attack_40_iters_acc_83.960.pth.tar"
#dataset path
data_path="/home/${USER}/data/pytorch/cifar10"
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../../../../ nohup $PYTHON main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${label_info} \
    --epochs ${epochs} --learning_rate 0.1 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --attack_eval --attack 'pgd' --attack_iters 40 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt

PYTHON="/home/${USER}/anaconda3/bin/python" # python environment
enable_tb_display=false # enable tensorboard display
model=noise_resnet20_weight # + adv. training
dataset=svhn
epochs=160
batch_size=2500
optimizer=SGD
# add more labels as additional info into the saving path
label_info=train_layerwise_3e-4decay_adv_eval
pretrained_model="/home/${USER}/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/svhn_pni_weight.pth.tar"
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
    --resume ${pretrained_model} \
    --attack_eval --attack 'cw' --attack_iters 200 \
    --epoch_delay 5 >> test_${timestamp}.txt 2>&1 &
echo test_${timestamp}.txt
[1] 33831
cc@p:~/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code$ echo test_${timestamp}.txt
test_2020-02-10-19-20-35-574208044.txt


