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
from cnns.nnlib.attacks.carlini_wagner_round import CarliniWagnerL2AttackRound
from cnns.nnlib.robustness.utils import get_foolbox_model
from cnns.nnlib.datasets.cifar import cifar_min
from cnns.nnlib.datasets.cifar import cifar_max
from cnns.nnlib.datasets.cifar import cifar_mean
from cnns.nnlib.datasets.cifar import cifar_std
import socket
from cnns.nnlib.utils.general_utils import get_log_time
import os
from cnns.nnlib.datasets.transformations.denorm_round_norm import \
    DenormRoundNorm
from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance
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
        # (foolbox.attacks.CarliniWagnerL2Attack, "CarliniWagnerL2Attack",
        # [x for x in range(2, 20, 1)] + [x for x in
        #                                 range(20, 1000, 10)] + [1000]),
        # [1000]),
        # [2]),
        (CarliniWagnerL2AttackRound, "CarliniWagnerL2AttackRound",
         [x for x in range(1000)]),
    ]
    return attacks


# attack = foolbox.attacks.FGSM(model)
# attack = empty_attack

def run(args):
    import datetime
    print("start time: ", datetime.datetime.now())

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
        "corrected by band",
        "mean original distance",
        "mean rounded distance",
        "original L2 distance",
        "rounded L2 distance",
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
        # (84,  # FFT based modle
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

    # full_model_path = "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"
    full_model_spectra_path = "full_spectra.model"
    full_model_path = "2019-04-08-18-32-59-787750-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.47.model"
    band_model_path = "2019-01-21-14-30-13-992591-dataset-cifar10-preserve-energy-100.0-test-accuracy-84.55-compress-label-84-after-epoch-304.model"
    args.conv_type = ConvType.STANDARD2D
    full_model = get_foolbox_model(args, model_path=full_model_path,
                                   compress_rate=0,
                                   min=cifar_min, max=cifar_max)
    args.conv_type = ConvType.FFT2D
    full_model_spectra = get_foolbox_model(args,
                                           model_path=full_model_spectra_path,
                                           compress_rate=0,
                                           min=cifar_min, max=cifar_max)
    args.conv_type = ConvType.FFT2D
    band_model = get_foolbox_model(args, model_path=band_model_path,
                                   compress_rate=84,
                                   min=cifar_min, max=cifar_max)

    round_model = band_model

    # full_attack = CarliniWagnerL2AttackRound(full_model)
    full_attack = CarliniWagnerL2AttackRound(round_model)
    # input_epsilons = [1000]
    input_epsilons = range(1000)
    values_per_channel = 256

    distance_measure = DenormDistance(mean=cifar_mean, std=cifar_std)

    # for epsilon in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
    for epsilon in input_epsilons:
        original_L2_distance = 0.0
        rounded_L2_distance = 0.0
        original_distance = 0.0
        rounded_distance = 0.0
        no_adversarials = 0  # no adversarial found
        adversarials = 0
        correct_round = 0  # corrected by round
        correct_band = 0  # corrected by fft band limited model
        counter = 0  # count of correctly classified images
        total_counter = 0  # total counter of images
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

                # check if the image is classified correctly by both models
                predictions = band_model.predictions(model_image)
                if np.argmax(predictions) != label:
                    # print("not classified correctly")
                    continue
                predictions = full_model.predictions(model_image)
                if np.argmax(predictions) != label:
                    # print("not classified correctly")
                    continue

                counter += 1
                original_adversarial, rounded_adversarial = full_attack(
                    image, label, max_iterations=epsilon, abort_early=False,
                    unpack=False, values_per_channel=values_per_channel)
                original_image_attack = original_adversarial.image
                rounded_image_attack = rounded_adversarial.image
                # print("original distance: ", original_adversarial.distance.value)
                if original_adversarial.distance.value < np.inf:
                    original_distance += original_adversarial.distance.value
                if rounded_adversarial.distance.value < np.inf:
                    rounded_distance += rounded_adversarial.distance.value
                if original_image_attack is not None:
                    original_L2_distance += distance_measure.measure(image,
                                                                     original_image_attack)
                if rounded_image_attack is not None:
                    rounded_L2_distance += distance_measure.measure(image,
                                                                    rounded_image_attack)

                if rounded_image_attack is None:
                    no_adversarials += 1
                    # print("image is None, label:", label, " i:", i)
                elif args.is_round:
                    # print("batch idx: ", batch_idx, " image idx: ", i,
                    #       " label: ", label)
                    adversarials += 1
                    # print("sum difference before round: ",
                    #       np.sum(
                    #           np.abs(image_attack * 255 - image * 255)))
                    # image_attack = np.round(image_attack * 255) / 255
                    # for values_per_channel in [256 // (2 ** x) for x in
                    #                            range(0, 7)]:
                    rounded_image_attack = DenormRoundNorm(
                        mean=cifar_mean, std=cifar_std,
                        values_per_channel=values_per_channel).round(
                        rounded_image_attack)
                    # predictions = full_model.predictions(rounded_image_attack)
                    predictions = round_model.predictions(rounded_image_attack)

                    # print(np.argmax(predictions), label)
                    if np.argmax(predictions) == label:
                        correct_round += 1
                        # print(",".join([str(x) for x in [
                        #     "epsilon", epsilon,
                        #     "batch_idx", batch_idx,
                        #     "i", i,
                        #     "values per channel",
                        #     values_per_channel]]))
                    else:
                        predictions = band_model.predictions(
                            rounded_image_attack)
                        if np.argmax(predictions) == label:
                            correct_band += 1
        timing = time.time() - start
        with open(args.out_file_name, "a") as out:
            if adversarials > 0:
                corrected_round_rate = correct_round / adversarials * 100
                mean_original_distance = original_distance / adversarials
                mean_rounded_distance = rounded_distance / adversarials
                mean_original_L2_distance = original_L2_distance / adversarials
                mean_rounded_L2_distance = rounded_L2_distance / adversarials
            else:
                corrected_round_rate = 0.0
                mean_original_distance = 0.0
                mean_rounded_distance = 0.0
            msg = ",".join((str(x) for x in
                            ["0 and 84",
                             "CarliniWagnerL2Round",
                             epsilon,
                             total_counter,
                             no_adversarials,
                             counter,
                             no_adversarials / counter * 100,
                             correct_round,
                             adversarials,
                             corrected_round_rate,
                             correct_band,
                             mean_original_distance,
                             mean_rounded_distance,
                             mean_original_L2_distance,
                             mean_rounded_L2_distance,
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
    # for model with rounding

    # args.model_path = "2019-01-21-14-30-13-992591-dataset-cifar10-preserve-energy-100.0-test-accuracy-84.55-compress-label-84-after-epoch-304.model"
    args.values_per_channel = 0
    # args.compress_rate = 84
    args.conv_type = ConvType.FFT2D
    args.sample_count_limit = 100

    # args.model_path = "2019-04-08-19-53-50-779103-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.48-rounding-32-values-per-channel.model"
    # args.model_path = "saved_model_2019-04-11-04-51-57-429818-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.48-channel-vals-256.model"
    # args.values_per_channel = 256
    # args.conv_type = ConvType.STANDARD2D
    # args.sample_count_limit = 100
    # args.model_path = "saved_model_2019-04-13-07-22-56-806744-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.23-channel-vals-256.model"
    # args.model_path = "2019-04-08-19-53-50-779103-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.48-rounding-32-values-per-channel.model
    # args.model_path = "saved_model_2019-04-11-07-18-28-194468-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-78.25-channel-vals-2.model"
    # args.values_per_channel = 2

    # args.conv_type = ConvType.STANDARD2D
    args.dataset = "cifar10"
    args.network_type = NetworkType.ResNet18
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
