# Band-limited Training and Inference for Convolutional Neural Networks

The convolutional layers are core building blocks of neural network architectures. In general, a convolutional filter applies to the entire frequency spectrum of an input signal. We explore artificially constraining the frequency spectra of these filters, called band-limiting, during Convolutional Neural Networks (CNN) training. The band-limiting applies to both the feedforward and backpropagation steps. Through an extensive evaluation over time-series and image datasets, we observe that CNNs are resilient to this compression scheme and results suggest that CNNs learn to leverage lower-frequency components. An extensive experimental evaluation across 1D and 2D CNN training tasks illustrates: (1) band-limited training can effectively control the resource usage (GPU and memory); (2) models trained with band-limited layers retain high prediction accuracy; and (3) requires no modification to existing training algorithms or neural network architectures to use unlike other compression schemes.

# Installation
See guidelines in: `cnns/installations`
- Pytorch version 1.0 (recommended installation from the source code: pytorch_from_source.sh)
- Torchvision (recommended from the source code: https://github.com/pytorch/vision.git, torchvision.sh)
- Apex: https://github.com/NVIDIA/apex (nvidia_apex.sh)

Install python libraries:
- matplotlib
- memory_profiler
- py3nvml
- torch_dct (pip install torch_dct)

Install CUDA kernel:
`cnns/nnlib/pytorch_cuda/complex_mul_cuda/clean_install.sh` (adjust your python path inside the bash script).

We ran the experiments with CUDA 9.2.

# Experiments
To run the main experiments for cifar10 and cifar100, go to the directory: `cnns/nnlib/pytorch_experiments`.
Find the `run.sh` file and run some examples from it. This is how to train 2 exemplar models:

We assume that your Python installation is in: `/home/${USER}/anaconda3/bin/python3.6`, please change it to your Python path.
The file to run the experiments is `main.py`.

For CIFAR-10 with more than about 50% compression (label 48 or above for the compress_rates param):
`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main.py --adam_beta2=0.999 --compress_type='STANDARD' --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar10' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=350 --compress_rates 48 --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='STANDARD' --min_batch_size=32 --model_path='no_model' --momentum=0.9 --network_type='ResNet18' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies 100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='FALSE' --weight_decay=0.0005 --workers=6 --precision_type='FP32'  --only_train='FALSE' --test_compress_rate='FALSE' --noise_sigmas=0.0 >> cifar10-fft2d-energy100-pytorch-adam-gpu-lr:0.01,decay:0.0005-compress-rate-48.0-percent-float32.txt 2>&1 &`

For CIFAR-100 with more than about 50% compression (label 48 or above for the compress_rates param):
`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../../ nohup /home/${USER}/anaconda3/bin/python3.6 main.py --adam_beta2=0.999 --compress_type='STANDARD' --conv_type='FFT2D' --conv_exec_type=CUDA --dataset='cifar100' --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=350 --compress_rates 48  --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='STANDARD' --min_batch_size=32 --model_path='no_model' --momentum=0.9 --network_type='DenseNetCifar' --next_power2='TRUE' --optimizer_type='MOMENTUM' --preserve_energies 100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='FALSE' --weight_decay=0.0001 --workers=6 --precision_type='FP32'  --only_train='FALSE' --test_compress_rate='FALSE' --noise_sigmas=0.0 >> cifar100-fft2d-energy100-pytorch-adam-gpu-lr:0.01,decay:0.0001-compress-rate-12.0-percent-float32.txt 2>&1 &`

Please, feel free to modify the compression rate.

If you want to run the code inside your favorite IDE and want to have access to the program parameters (called args), then please go to the following file: `cnns/nnlib/utils/arguments.py`. The description of the arguments is in `cnns/nnlib/utils/exec_args.py`.

Most graphs from the paper can be found in: `cnns/graphs` directory.

# Implementation:
The main part of the 2D convolution with compression can be found in the following files:
1. Pytorch conv2D_fft module: `cnns/nnlib/pytorch_layers/conv2D_fft.py`
2. CUDA kernel: `cnns/nnlib/pytorch_cuda/complex_mul_cuda/complex_mul_kernel_stride_no_permute.cu`
3. Pytorch conv1D_fft module: `cnns/nnlib/pytorch_layers/conv1D_fft.py`

## Memory management:
`cnns/pytorch_tutorials/memory_net.py`

Find cliffs in the execution of neural networks.
Go to the level of C++ and cuda.
Find for what input size, the memory size is not sufficient.
Run a single forward pass and a subsequent backward pass.

Define neural network, compute loss and make updates to the weights of the
network.


We do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. ExperimentSpectralSpatial the network on the test data
