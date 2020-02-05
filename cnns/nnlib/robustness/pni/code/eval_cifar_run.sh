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

