#  Band-limited CNNs
#  Copyright (c) 2019. Adam Dziedzic
#  Licensed under The Apache License [see LICENSE for details]
#  Written by Adam Dziedzic
from cnns import matplotlib_backend

print("Using:", matplotlib_backend.backend)

from cnns.nnlib.utils.general_utils import ConvType
import foolbox
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.cifar import get_cifar
import torch
import sys
import time
import numpy as np
from cnns.nnlib.robustness.utils import get_foolbox_model
# from cnns.nnlib.robustness.utils import get_min_max_counter
from cnns.nnlib.datasets.cifar import cifar_min
from cnns.nnlib.datasets.cifar import cifar_max
from cnns.nnlib.datasets.cifar import cifar_mean
from cnns.nnlib.datasets.cifar import cifar_std
import socket
from cnns.nnlib.utils.general_utils import get_log_time
import os
from cnns.nnlib.datasets.transformations.denorm_round_norm import \
    DenormRoundNorm
from cnns.nnlib.utils.general_utils import NetworkType

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


class empty_attack(foolbox.attacks.base.Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        return input_or_adv


def get_attacks():
    attacks = [  # empty_attack,
        # (foolbox.attacks.LocalSearchAttack, "LocalSearch",
        #  [x for x in range(1000)]),
        # (foolbox.attacks.MultiplePixelsAttack, "MultiplePixelsAttack",
        # [x for x in range(100, 301, 10)]),
        # (foolbox.attacks.SinglePixelAttack, "SinglePixelAttack",
        #  [x for x in range(0, 1001, 100)]),
        # # foolbox.attacks.AdditiveUniformNoiseAttack,
        # foolbox.attacks.GaussianBlurAttack,
        # foolbox.attacks.AdditiveGaussianNoiseAttack,
        # foolbox.attacks.FGSM,
        # (foolbox.attacks.ContrastReductionAttack, [0.8 + x / 10 for x in range(3)]),
        # (foolbox.attacks.SpatialAttack, "Rotations",
        #  [x for x in range(0, 21, 1)]),
        # (foolbox.attacks.SpatialAttack, "Translations",
        #  [x for x in range(0, 21, 1)]),
        # (foolbox.attacks.SpatialAttack, "All", [x for x in range(0, 21, 1)]),
        # (foolbox.attacks.ContrastReductionAttack, "ContrastReductionAttack",
        #  [x / 10 for x in range(10,0,-1)]),
        # (foolbox.attacks.GradientAttack, [x / 100 for x in range(21)]),
        # (foolbox.attacks.GradientSignAttack, "GradientSignAttack",
        #  [x for x in np.linspace(0.001, 0.2, 20)][1:]),
        # (
        #     foolbox.attacks.BlendedUniformNoiseAttack,
        #     [x / 10 for x in range(21)]),
        # foolbox.attacks.SaltAndPepprunerNoiseAttack(foolbox_model),
        # foolbox.attacks.LinfinityBasicIterativeAttack(
        # model, distance=foolbox.distances.MeanSquaredDistance),
        (foolbox.attacks.CarliniWagnerL2Attack, "CarliniWagnerL2Attack",
        # [x for x in range(2, 20, 1)] + [x for x in
        #                                 range(20, 1000, 10)] + [1000]),
        # [1000]),
        [2]),
    ]
    return attacks


# attack = foolbox.attacks.FGSM(model)
# attack = empty_attack

def run(args):
    header = ",".join([
        "compress rate",
        "attack name",
        "epsilon",
        "total counter",
        "# no adversarials found",
        "# correctly classified",
        "no adversarial rate (%)",
        "# corrected by round",
        "# adversarials",
        "corrected by round rate (%)",
        "time (sec)"])
    print(header)
    with open(args.out_file_name, "a") as out:
        out.write(header + "\n")

    model_paths = [
        # (0,  # standard 2D conv model with 32 values per channel
        #  # "2019-04-08-12-01-38-163679-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-87.53.model"),
        #  "2019-04-08-14-21-46-099982-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-88.41.model"),
        # (0,  # FFT based model
        #  "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"),
        # (84,  # FFT based model
        #  "2019-01-21-14-30-13-992591-dataset-cifar10-preserve-energy-100.0-test-accuracy-84.55-compress-label-84-after-epoch-304.model"),
        # (0,
        #  "2019-04-08-19-53-50-779103-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.48-rounding-32-values-per-channel.model")
    ]

    # input_epsilons = [0.7, 0.8, 0.9, 1.0]
    # input_epsilons = [0.01, 0.1, 1.0]
    # input_epsilons = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # 0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02
    # input_epsilons = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    # input_epsilons = [0.11 + x / 100 for x in range(10)]
    # input_epsilons = [x / 100 for x in range(21)]
    # input_epsilons = range(0, 100, 10)

    import datetime

    print("start time: ", datetime.datetime.now())

    attacks = get_attacks()

    for current_attack, attack_type, input_epsilons in attacks:
        # for compress_rate, model_path in model_paths:
        for compress_rate, model_path in [
            (args.compress_rate, args.model_path)]:
            print('model path: ', model_path)
            foolbox_model = get_foolbox_model(args, model_path=model_path,
                                              compress_rate=compress_rate,
                                              min=cifar_min, max=cifar_max)
            attack = current_attack(foolbox_model)
            print("attack type: ", attack_type)
            # for epsilon in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
            for epsilon in input_epsilons:
                if attack.name() == "SaltAndPepperNoiseAttack":
                    epsilon = int(epsilon * 100)
                    epsilons = epsilon
                else:
                    epsilons = [epsilon]
                no_adversarials = 0  # no adversarial found
                adversarials = 0
                correct_round = 0  # corrected by round
                counter = 0  # total count of correctly classified images
                total_counter = 0  # total counter of checked images
                start = time.time()
                for batch_idx, (data, target) in enumerate(test_loader):
                    # print("batch_idx: ", batch_idx)
                    for i, label in enumerate(target):
                        total_counter += 1
                        label = label.item()
                        image = data[i].numpy()
                        # the image has to be classified correctly in the first
                        # place
                        model_image = image
                        # if args.values_per_channel > 0:
                        #     model_image = RoundingTransformation(
                        #         values_per_channel=args.values_per_channel,
                        #         round=np.round)(model_image)
                        predictions = foolbox_model.predictions(
                            model_image)
                        if np.argmax(predictions) != label:
                            # print("not classified correctly")
                            continue
                        counter += 1
                        if attack.name() == "CarliniWagnerL2Attack":
                            image_attack = attack(image, label,
                                                  max_iterations=epsilon,
                                                  abort_early=False)
                        elif attack_type == "Rotations":
                            image_attack = attack(image, label,
                                                  do_rotations=True,
                                                  do_translations=False,
                                                  x_shift_limits=(
                                                      -epsilon, epsilon),
                                                  y_shift_limits=(
                                                      -epsilon, epsilon),
                                                  angular_limits=(
                                                      -epsilon, epsilon),
                                                  granularity=10,
                                                  random_sampling=False,
                                                  abort_early=True)
                        elif attack_type == "Translations":
                            image_attack = attack(image, label,
                                                  do_rotations=False,
                                                  do_translations=True,
                                                  x_shift_limits=(
                                                      -epsilon, epsilon),
                                                  y_shift_limits=(
                                                      -epsilon, epsilon),
                                                  angular_limits=(
                                                      -epsilon, epsilon),
                                                  granularity=10,
                                                  random_sampling=False,
                                                  abort_early=True)
                        elif attack.name() == "SpatialAttack":
                            image_attack = attack(image, label,
                                                  do_rotations=True,
                                                  do_translations=True,
                                                  x_shift_limits=(
                                                      -epsilon, epsilon),
                                                  y_shift_limits=(
                                                      -epsilon, epsilon),
                                                  angular_limits=(
                                                      -epsilon, epsilon),
                                                  granularity=10,
                                                  random_sampling=False,
                                                  abort_early=True)
                        elif attack_type == "SinglePixelAttack":
                            image_attack = attack(image, label,
                                                  max_pixels=epsilon,
                                                  pixel_type="single")
                        elif attack_type == "MultiplePixelsAttack":
                            image_attack = attack(image, label,
                                                  num_pixels=epsilon)
                        elif attack.name() == "LocalSearchAttack":
                            image_attack = attack(image, label, t=epsilon)
                        else:
                            image_attack = attack(image, label,
                                                  epsilons=epsilons)
                        if image_attack is None:
                            no_adversarials += 1
                            # print("image is None, label:", label, " i:", i)
                        elif args.is_round:
                            print("batch idx: ", batch_idx, " image idx: ", i,
                                  " label: ", label)
                            adversarials += 1
                            # print("sum difference before round: ",
                            #       np.sum(
                            #           np.abs(image_attack * 255 - image * 255)))
                            # image_attack = np.round(image_attack * 255) / 255
                            # for values_per_channel in [256 // (2 ** x) for x in
                            #                            range(0, 7)]:
                            for values_per_channel in [args.values_per_channel]:
                                image_attack_torch = torch.from_numpy(
                                    image_attack)
                                image_attack = DenormRoundNorm(
                                    mean=cifar_mean, std=cifar_std,
                                    values_per_channel=values_per_channel)(
                                    image_attack_torch).numpy()
                                predictions = foolbox_model.predictions(
                                    image_attack)

                                # print(np.argmax(predictions), label)
                                if np.argmax(predictions) == label:
                                    correct_round += 1
                                    print(",".join([str(x) for x in [
                                        "epsilon", epsilon,
                                        "batch_idx", batch_idx,
                                        "i", i,
                                        "values per channel",
                                        values_per_channel]]))
                                    break
                timing = time.time() - start
                with open(args.out_file_name, "a") as out:
                    if adversarials > 0:
                        corrected_round_rate = correct_round / adversarials * 100
                    else:
                        corrected_round_rate = 100.0
                    msg = ",".join((str(x) for x in
                                    [compress_rate,
                                     attack.name(),
                                     epsilon,
                                     total_counter,
                                     no_adversarials,
                                     counter,
                                     no_adversarials / counter * 100,
                                     correct_round,
                                     adversarials,
                                     corrected_round_rate,
                                     timing]))
                    print(msg)
                    out.write(msg + "\n")

    print("end time: ", datetime.datetime.now())


if __name__ == "__main__":
    np.random.seed(31)
    args = get_args()
    # should we turn pixels to the range from 0 to 255 and round them to
    # the nearest integer values?
    args.is_round = True
    args.dataset = "cifar10"
    args.network_type = NetworkType.ResNet18
    # for model with rounding

    args.model_path = "2019-01-21-14-30-13-992591-dataset-cifar10-preserve-energy-100.0-test-accuracy-84.55-compress-label-84-after-epoch-304.model"
    args.values_per_channel = 0
    args.compress_rate = 84
    args.conv_type = ConvType.FFT2D
    args.sample_count_limit = 100

    # args.model_path = "2019-04-08-19-53-50-779103-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.48-rounding-32-values-per-channel.model"
    # args.model_path = "saved_model_2019-04-11-04-51-57-429818-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.48-channel-vals-256.model"
    # args.values_per_channel = 256
    # args.compress_rate = 0
    # args.conv_type = ConvType.STANDARD2D
    # args.sample_count_limit = 100

    # args.model_path = "saved_model_2019-04-13-07-22-56-806744-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.23-channel-vals-256.model"
    # args.model_path = "2019-04-08-19-53-50-779103-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.48-rounding-32-values-per-channel.model
    # args.model_path = "saved_model_2019-04-11-07-18-28-194468-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-78.25-channel-vals-2.model"
    # args.values_per_channel = 2

    # args.conv_type = ConvType.STANDARD2D

    train_loader, test_loader, train_dataset, test_dataset = get_cifar(
        args=args, dataset_name=args.dataset)

    if torch.cuda.is_available() and args.use_cuda:
        print("cuda is available")
        args.device = torch.device("cuda")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("cuda id not available")
        args.device = torch.device("cpu")

    # min, max, counter = get_min_max_counter(test_loader=test_loader)
    # print("counter: ", counter, " min: ", min, " max: ", max)
    hostname = socket.gethostname()
    try:
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        cuda_visible_devices = 0
    args_str = args.get_str()
    HEADER = "hostname," + str(
        hostname) + ",timestamp," + get_log_time() + "," + str(
        args_str) + ",cuda_visible_devices," + str(cuda_visible_devices)
    args.out_file_name = get_log_time() + os.path.basename(
        __file__) + "_" + args.dataset + ".csv"
    with open(args.out_file_name, "a") as file:  # Write the metadata.
        file.write(HEADER + "\n")

    print(args.get_str())
    run(args)
