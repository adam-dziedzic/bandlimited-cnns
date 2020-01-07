from cnns import matplotlib_backend

print('Using: ', matplotlib_backend.backend)

import matplotlib

print('Using: ', matplotlib.get_backend())

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

import torch
import time
import numpy as np
from cnns.nnlib.utils.exec_args import get_args
import cnns.foolbox.foolbox_2_3_0 as foolbox
from cnns.nnlib.robustness.pytorch_model import get_model
from cnns.nnlib.robustness.utils import gauss_noise
from cnns.nnlib.robustness.param_perturbation.utils import get_data_loader
from cnns import matplotlib_backend
import sys


def get_perturbed_fmodel(args):
    fmodel = get_fmodel(args)
    model = fmodel._model
    params = model.parameters()
    with torch.no_grad():
        for param in params:
            shape = list(param.shape)
            noise = gauss_noise(epsilon=args.noise_sigma, args=args,
                                shape=shape, dtype=np.float)
            noise = torch.tensor(noise, dtype=param.dtype, device=param.device)
            param.data += noise
    return fmodel


def get_fmodel(args):
    pytorch_model = get_model(args)
    # preprocessing = dict(mean=args.mean_array,
    #                      std=args.std_array,
    #                      axis=-3)
    fmodel = foolbox.models.PyTorchModel(pytorch_model,
                                         bounds=(args.min, args.max),
                                         channel_axis=1,
                                         device=args.device,
                                         num_classes=args.num_classes,
                                         preprocessing=(0, 1))
    return fmodel


def get_accuracy(fmodel, data_loader):
    total_count = 0
    predict_count = 0

    for batch_idx, (images, labels) in enumerate(data_loader):
        total_count += len(labels)
        images, labels = images.numpy(), labels.numpy()
        # print('labels: ', labels)

        predict_labels = fmodel.forward(images).argmax(axis=-1)
        predict_count += np.sum(predict_labels == labels)
        # print('accuracy: ', predict_count / total_count)
    return predict_count / total_count


def compute(args):
    data_loader = get_data_loader(args)

    start = time.time()
    clean_fmodel = get_fmodel(args)
    clean_accuracy = get_accuracy(fmodel=clean_fmodel, data_loader=data_loader)
    print(f'clean {args.use_set} accuracy: ', clean_accuracy)
    print('elapsed time: ', time.time() - start)

    print(f'noise sigma, perturb {args.use_set} accuracy, elapsed time')

    # for noise_sigma in args.noise_sigmas:
    for noise_sigma in np.linspace(0.0001, 0.01, 1000):
        start = time.time()
        args.noise_sigma = noise_sigma
        perturb_fmodel = get_perturbed_fmodel(args)
        perturb_accuracy = get_accuracy(fmodel=perturb_fmodel,
                                        data_loader=data_loader)
        elapsed_time = time.time() - start
        print(args.noise_sigma, ',', perturb_accuracy, ',', elapsed_time)
        sys.stdout.flush()


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    compute(args)
    print("total elapsed time: ", time.time() - start_time)
