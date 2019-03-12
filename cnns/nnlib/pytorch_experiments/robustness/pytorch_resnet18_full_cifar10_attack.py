#  Band-limited CNNs
#  Copyright (c) 2019. Adam Dziedzic
#  Licensed under The Apache License [see LICENSE for details]
#  Written by Adam Dziedzic

import foolbox
import numpy as np
from cnns.nnlib.pytorch_architecture.resnet2d import resnet18
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.cifar import get_cifar
import torch
import sys
import os
import time

np.random.seed(31)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

args = get_args()
train_loader, test_loader, train_dataset, test_dataset = get_cifar(args, "cifar10")

if torch.cuda.is_available() and args.use_cuda:
    print("cuda is available")
    device = torch.device("cuda")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("cuda id not available")
    device = torch.device("cpu")

min = float("inf")
max = float("-inf")

counter = 0
for batch_idx, (data, target) in enumerate(test_loader):
    # print("batch_idx: ", batch_idx)
    for i, label in enumerate(target):
        counter += 1
        label = label.item()
        image = data[i].numpy()
        if image.min() < min:
            min = image.min()
        if image.max() > max:
            max = image.max()
print("counter: ", counter, " min: ", min, " max: ", max)


def load_model(args):
    model = resnet18(args=args)
    # load pretrained weights
    models_folder_name = "models"
    models_dir = os.path.join(os.getcwd(), os.path.pardir, models_folder_name)
    if args.model_path != "no_model":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=device))
        msg = "loaded model: " + args.model_path
        # logger.info(msg)
        print(msg)
    return model.eval()


class empty_attack(foolbox.attacks.base.Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        return input_or_adv


def get_foolbox_model(model_path, compress_rate):
    args.model_path = model_path
    args.compress_rate = compress_rate
    args.compress_rates = [compress_rate]
    pytorch_model = load_model(args=args)
    mean = 0
    std = 1
    foolbox_model = foolbox.models.PyTorchModel(model=pytorch_model, bounds=(min, max),
                                                num_classes=args.num_classes,
                                                preprocessing=(mean, std), device=device)
    return foolbox_model


def get_attacks():
    attacks = [  # empty_attack,
        # foolbox.attacks.SinglePixelAttack(model),
        # foolbox.attacks.AdditiveUniformNoiseAttack,
        # foolbox.attacks.GaussianBlurAttack,
        # foolbox.attacks.AdditiveGaussianNoiseAttack,
        foolbox.attacks.FGSM,
        foolbox.attacks.GradientAttack,
        foolbox.attacks.ContrastReductionAttack,
        foolbox.attacks.BlendedUniformNoiseAttack,
        # foolbox.attacks.SaltAndPepperNoiseAttack(foolbox_model),
        # foolbox.attacks.LinfinityBasicIterativeAttack(
        # model, distance=foolbox.distances.MeanSquaredDistance),
    ]
    return attacks


# attack = foolbox.attacks.FGSM(model)
# attack = empty_attack

print("compress rate, attack name, epsilon, correct, counter, correct rate (%), time (sec)")

model_paths = [
    (0,
     "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"),
    (84,
     "2019-01-21-14-30-13-992591-dataset-cifar10-preserve-energy-100.0-test-accuracy-84.55-compress-label-84-after-epoch-304.model"),

]

# input_epsilons = [0.7, 0.8, 0.9, 1.0]
# input_epsilons = [0.01, 0.1, 1.0]
# input_epsilons = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# 0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02
input_epsilons = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# input_epsilons = range(0, 100, 10)
attacks = get_attacks()
for current_attack in attacks:
    for compress_rate, model_path in model_paths:
        foolbox_model = get_foolbox_model(model_path=model_path, compress_rate=compress_rate)
        attack = current_attack(foolbox_model)
        # for epsilon in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        for epsilon in input_epsilons:
            if attack.name() == "SaltAndPepperNoiseAttack":
                epsilon = int(epsilon * 100)
                epsilons = epsilon
            else:
                epsilons = [epsilon]
            correct = 0
            counter = 0
            start = time.time()
            for batch_idx, (data, target) in enumerate(test_loader):
                # print("batch_idx: ", batch_idx)
                for i, label in enumerate(target):
                    counter += 1
                    label = label.item()
                    image = data[i].numpy()
                    if attack.name() == "SinglePixelAttack":
                        image_attack = attack(image, label, max_pixels=epsilon)
                    else:
                        image_attack = attack(image, label, epsilons=epsilons)
                    if image_attack is None:
                        correct += 1
                        # print("image is None, label:", label, " i:", i)
                    # else:
                    #     predictions = foolbox_model.predictions(image_attack)
                    #     # print(np.argmax(predictions), label)
                    #     if np.argmax(predictions) == label:
                    #         correct += 1
            timing = time.time() - start
            print(compress_rate, ",", attack.name(), ",", epsilon, ",", correct, ",", counter, ",", correct / counter,
                  ",", timing)
