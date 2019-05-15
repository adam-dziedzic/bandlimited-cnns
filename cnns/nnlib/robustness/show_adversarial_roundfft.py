#  Band-limiting
#  Copyright (c) 2019. Adam Dziedzic
#  Licensed under The Apache License [see LICENSE for details]
#  Written by Adam Dziedzic

"""
Use the rounding and fft pre-processing and find adversarial examples after
such transformations.
"""

# If you use Jupyter notebooks uncomment the line below
# %matplotlib inline

# Use the import below to run the code remotely on a server.
from cnns import matplotlib_backend

print("Using:", matplotlib_backend.backend)

import time
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import foolbox
import numpy as np
import torch
import torchvision.models as models
# from cnns.nnlib.pytorch_layers.pytorch_utils import get_full_energy
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from cnns.nnlib.pytorch_layers.pytorch_utils import get_phase
from cnns.nnlib.datasets.imagenet.imagenet_from_class_idx_to_label import \
    imagenet_from_class_idx_to_label
from cnns.nnlib.datasets.cifar10_from_class_idx_to_label import \
    cifar10_from_class_idx_to_label
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.cifar import get_cifar
# from scipy.special import softmax
from cnns.nnlib.robustness.utils import load_model
from cnns.nnlib.datasets.cifar import cifar_max, cifar_min
from cnns.nnlib.datasets.cifar import cifar_std_array, cifar_mean_array
from cnns.nnlib.datasets.mnist.mnist import mnist_max, mnist_min
from cnns.nnlib.datasets.mnist.mnist import mnist_std_array, mnist_mean_array
from cnns.nnlib.datasets.mnist.mnist_from_class_idx_to_label import \
    mnist_from_class_idx_to_label
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_max
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_min
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_mean_array
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_std_array
from cnns.nnlib.attacks.carlini_wagner_round_fft import \
    CarliniWagnerL2AttackRoundFFT
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.datasets.transformations.denorm_round_norm import \
    DenormRoundNorm
from cnns.nnlib.datasets.transformations.normalize import Normalize
from cnns.nnlib.datasets.transformations.denormalize import Denormalize
from cnns.nnlib.utils.general_utils import NetworkType
from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet
from cnns.nnlib.utils.complex_mask import get_disk_mask
from cnns.nnlib.utils.complex_mask import get_hyper_mask
from cnns.nnlib.utils.general_utils import AttackType


def softmax(x):
    s = np.exp(x - np.max(x))
    s /= np.sum(s)
    return s


def softmax_from_torch(x):
    s = torch.nn.functional.softmax(torch.tensor(x, dtype=torch.float))
    return s.numpy()


def to_fft(x, fft_type, is_log=True):
    x = torch.from_numpy(x)
    # x = torch.tensor(x)
    # x = x.permute(2, 0, 1)  # move channel as the first dimension
    xfft = torch.rfft(x, onesided=False, signal_ndim=2)
    if fft_type == "magnitude":
        return to_fft_magnitude(xfft, is_log)
    elif fft_type == "phase":
        return to_fft_phase(xfft)
    else:
        raise Exception(f"Unknown type of fft processing: {fft_type}")


def to_fft_magnitude(xfft, is_log=True):
    """
    Get the magnitude component of the fft-ed signal.

    :param xfft: the fft-ed signal
    :param is_log: for the logarithmic scale follow the dB (decibel) notation
    where ydb = 20 * log_10(y), according to:
    https://www.mathworks.com/help/signal/ref/mag2db.html
    :return: the magnitude component of the fft-ed signal
    """
    # _, xfft_squared = get_full_energy(xfft)
    # xfft_abs = torch.sqrt(xfft_squared)
    # xfft_abs = xfft_abs.sum(dim=0)
    xfft = get_spectrum(xfft)
    xfft = xfft.numpy()
    if is_log:
        # Ensure xfft does not have zeros.
        # xfft = xfft + 0.00001
        xfft = np.clip(xfft, 1e-12, None)
        xfft = 20 * np.log10(xfft)
        # print("xfft: ", xfft)
        # print("xfft min: ", xfft.min())
        # print("xfft max: ", xfft.max())
        return xfft
    else:
        return xfft


def to_fft_phase(xfft):
    # The phase is unwrapped using the unwrap function so that we can see a
    # continuous function of frequency.
    return np.unwrap(get_phase(xfft).numpy())


def znormalize(x):
    return (x - x.min()) / (x.max() - x.min())


def get_fmodel(args):
    if args.dataset == "imagenet":
        args.cmap = None
        args.init_y, args.init_x = 224, 224
        network_model = models.resnet50(
            pretrained=True).cuda().eval()  # for CPU, remove cuda()
        min = imagenet_min
        max = imagenet_max
        args.min = min
        args.max = max
        args.mean_array = imagenet_mean_array
        args.std_array = imagenet_std_array
        args.num_classes = 1000
        from_class_idx_to_label = imagenet_from_class_idx_to_label

    elif args.dataset == "cifar10":
        args.cmap = None
        args.init_y, args.init_x = 32, 32
        args.num_classes = 10
        # args.values_per_channel = 0
        # args.model_path = "saved_model_2019-04-08-16-51-16-845688-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.22-channel-vals-0.model"
        # args.model_path = "saved_model_2019-04-08-19-41-48-571492-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.5.model"
        # args.model_path = "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"
        # args.model_path = "saved_model_2019-04-13-06-54-15-810999-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-91.64-channel-vals-8.model"
        # args.compress_rate = 5
        # args.compress_rates = [args.compress_rate]
        # if args.model_path == "no_model":
        # args.model_path = "saved-model-2019-05-11-22-20-59-242197-dataset-cifar10-preserve-energy-100-compress-rate-5.0-test-accuracy-93.43-channel-vals-0.model"
        # args.attack_type = AttackType.BAND_ONLY
        # args.model_path = "saved_model_2019-04-13-10-25-49-315784-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.08-channel-vals-32.model"
        # args.model_path = "saved_model2019-05-11-18-54-18-392325-dataset-cifar10-preserve-energy-100.0-compress-rate-5.0-test-accuracy-91.21-channel-vals-8.model"
        args.in_channels = 3
        min = cifar_min
        max = cifar_max
        args.min = min
        args.max = max
        args.mean_array = cifar_mean_array
        args.std_array = cifar_std_array
        network_model = load_model(args=args)
        from_class_idx_to_label = cifar10_from_class_idx_to_label

    elif args.dataset == "mnist":
        args.cmap = "gray"
        args.init_y, args.init_x = 28, 28
        args.num_classes = 10
        # args.values_per_channel = 2
        # args.model_path = "2019-05-03-10-08-51-149612-dataset-mnist-preserve-energy-100-compress-rate-0.0-test-accuracy-99.07-channel-vals-0.model"
        # args.compress_rate = 0
        # args.compress_rates = [args.compress_rate]
        args.in_channels = 1
        min = mnist_min
        max = mnist_max
        args.min = min
        args.max = max
        args.mean_array = mnist_mean_array
        args.std_array = mnist_std_array
        # args.network_type = NetworkType.Net
        network_model = load_model(args=args)
        from_class_idx_to_label = mnist_from_class_idx_to_label

    else:
        raise Exception(f"Unknown dataset type: {args.dataset}")

    fmodel = foolbox.models.PyTorchModel(network_model, bounds=(min, max),
                                         num_classes=args.num_classes)

    return fmodel, from_class_idx_to_label


def run(args):
    fmodel, from_class_idx_to_label = get_fmodel(args=args)

    lim_y, lim_x = args.init_y, args.init_x
    # lim_y, lim_x = init_y // 2, init_x // 2
    # lim_y, lim_x = 2, 2
    # lim_y, lim_x = 3, 3

    cmap_type = "matshow"  # "standard" or "custom"
    # cmap_type = "standard"

    # vmin_heatmap = -6
    # vmax_heatmap = 10

    vmin_heatmap = None
    vmax_heatmap = None

    decimals = 4

    map_labels = "None"  # "None" or "Text"
    # map_labels = "Text"

    if cmap_type == "custom":
        # setting for the heat map
        # cdict = {
        #     'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
        #     'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
        #     'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
        # }

        cdict = {'red': [(0.0, 0.0, 0.0),
                         (0.5, 1.0, 1.0),
                         (1.0, 1.0, 1.0)],

                 'green': [(0.0, 0.0, 0.0),
                           (0.25, 0.0, 0.0),
                           (0.75, 1.0, 1.0),
                           (1.0, 1.0, 1.0)],

                 'blue': [(0.0, 0.0, 0.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 1.0, 1.0)]}

        # cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
        cmap = "OrRd"

        x = np.arange(0, lim_x, 1.)
        y = np.arange(0, lim_y, 1.)
        X, Y = np.meshgrid(x, y)
    elif cmap_type == "standard":
        # cmap = 'hot'
        cmap = 'OrRd'
        interpolation = 'nearest'
    elif cmap_type == "seismic":
        cmap = "seismic"
    elif cmap_type == "matshow":
        # cmap = "seismic"
        cmap = 'OrRd'
    else:
        raise Exception(f"Unknown type of the cmap: {cmap_type}.")

    def print_heat_map(input_map, args=args, title="", ylabel=""):
        args.plot_index += 1
        plt.subplot(rows, cols, args.plot_index)
        if args.plot_index % cols == 1:
            plt.ylabel(ylabel)
        plt.title(title)
        plt.imshow(input_map, cmap='hot', interpolation='nearest')
        # plt.axis('off')

    def print_color_map(x, fig=None, ax=None):
        if cmap_type == "standard":
            plt.imshow(x, cmap=cmap,
                       interpolation=interpolation)
            heatmap_legend = plt.pcolor(x)
            plt.colorbar(heatmap_legend)
        elif cmap_type == "custom":
            plt.pcolor(X, Y, x, cmap=cmap, vmin=vmin_heatmap,
                       vmax=vmax_heatmap)
            plt.colorbar()
        elif cmap_type == "matshow":
            # plt.pcolor(X, Y, original_fft, cmap=cmap, vmin=vmin_heatmap,
            #            vmax=vmax_heatmap)
            if vmin_heatmap != None:
                cax = ax.matshow(x, cmap=cmap, vmin=vmin_heatmap,
                                 vmax=vmax_heatmap)
            else:
                cax = ax.matshow(x, cmap=cmap)
            # plt.colorbar()
            fig.colorbar(cax)
            if map_labels == "Text":
                for (i, j), z in np.ndenumerate(x):
                    ax.text(j, i, str(np.around(z, decimals=decimals)),
                            ha='center', va='center')

    # Choose how many channels should be plotted.
    channels_nr = 1
    # Choose what fft types should be plotted.
    fft_types = ["magnitude"]
    # fft_types = ["magnitude", "phase"]
    # fft_types = []
    channels = [x for x in range(channels_nr)]
    attack_round_fft = CarliniWagnerL2AttackRoundFFT(model=fmodel, args=args,
                                                     get_mask=get_hyper_mask)
    # attacks = [
    #     # CarliniWagnerL2AttackRoundFFT(model=fmodel, args=args,
    #     #                               get_mask=get_hyper_mask),
    #     foolbox.attacks.CarliniWagnerL2Attack(fmodel),
    #     # foolbox.attacks.FGSM(fmodel),
    #     # foolbox.attacks.AdditiveUniformNoiseAttack(fmodel)
    # ]
    if args.attack_name == "CarliniWagnerL2Attack":
        attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
    elif args.attack_name == "CarliniWagnerL2AttackRoundFFT":
        attack = CarliniWagnerL2AttackRoundFFT(model=fmodel, args=args,
                                               get_mask=get_hyper_mask)
    else:
        raise Exception(f"Unknown attack name: {args.attack_name}")
    attacks = [attack]
    # 1 is for the first row of images.
    rows = len(attacks) * (1 + len(fft_types) * len(channels))
    cols = 1  # print at least the original image
    if args.values_per_channel > 0:
        cols += 1
    if args.compress_fft_layer > 0:
        cols += 1
    if args.is_adv_attack:
        cols += 1

    fig = plt.figure(figsize=(cols * 10, rows * 10))

    # index for each subplot
    args.plot_index = 0

    normalizer = Normalize(mean_array=args.mean_array,
                           std_array=args.std_array)
    denormalizer = Denormalize(mean_array=args.mean_array,
                               std_array=args.std_array)
    meter = DenormDistance(mean_array=args.mean_array,
                           std_array=args.std_array)

    if args.use_foolbox_data:
        images, labels = foolbox.utils.samples(dataset=args.dataset, index=0,
                                               batchsize=20,
                                               shape=(args.init_y, args.init_x),
                                               data_format='channels_first')
        print("max value in images pixels: ", np.max(images))
        images = images / 255
        print("max value in images after 255 division: ", np.max(images))
    elif args.dataset == "imagenet":
        train_loader, test_loader, train_dataset, test_dataset = load_imagenet(
            args)
    elif args.dataset == "cifar10":
        train_loader, test_loader, train_dataset, test_dataset = get_cifar(
            args, args.dataset)

    for attack in attacks:
        # get source image and label, args.idx - is the index of the image
        # image, label = foolbox.utils.imagenet_example()
        if args.use_foolbox_data:
            original_image, args.original_class_id = images[args.index], labels[
                args.index]
            original_image = original_image.astype(np.float32)

            # Normalize the data for the Pytorch models.
            original_image = normalizer.normalize(original_image)
        else:
            original_image, args.original_class_id = test_dataset.__getitem__(
                args.index)
            original_image = original_image.numpy()

        args.original_label = from_class_idx_to_label[args.original_class_id]
        print("original class id:", args.original_class_id, ", is label: ",
              args.original_label)

        def show_image(image, original_image, title="", args=args,
                       clip_input_image=True):
            original_class_id = args.original_class_id
            original_label = args.original_label
            predictions, _ = fmodel.predictions_and_gradient(
                image=image, label=original_class_id)
            # predictions_original = znormalize(predictions_original)
            predictions = softmax(predictions)
            predicted_class_id = np.argmax(predictions)
            predicted_label = from_class_idx_to_label[predicted_class_id]
            print(title)
            print("Number of unique values: ", len(np.unique(image)))
            print("model predicted label: ", predicted_label)
            if predicted_label != original_label:
                print(f"The original label: {original_label} is different than "
                      f"the predicted label: {predicted_label}")
            confidence = np.max(predictions)
            # sum_predictions = np.sum(predictions_original)
            # print("sum predictions: ", sum_predictions)
            # print("predictions_original: ", predictions_original)
            title_str = title + '\n'
            title_str += 'label: ' + str(
                predicted_label.replace(",", "\n")) + "\n"
            confidence_str = str(np.around(confidence, decimals=decimals))
            title_str += "confidence: " + confidence_str + "\n"
            L2_distance = meter.measure(original_image, image)
            if id(image) != id(original_image):
                title_str += "L2 distance: " + str(L2_distance) + "\n"
            ylabel_text = "spatial domain"
            if clip_input_image:
                # image = torch.clamp(image, min = args.min, max=args.max)
                image = np.clip(image, a_min=args.min, a_max=args.max)
            image_show = denormalizer.denormalize(image)
            if clip_input_image:
                image_show = np.clip(image, a_min=0, a_max=1)
            if args.dataset == "mnist":
                # image_show = image_show.astype('uint8')
                # plt.imshow(image_show.squeeze(), cmap=args.cmap,
                #            interpolation='nearest')
                print_heat_map(input_map=image_show[0], args=args,
                               ylabel=ylabel_text, title=title_str)
            else:
                args.plot_index += 1
                plt.subplot(rows, cols, args.plot_index)
                plt.title(title_str)
                plt.imshow(
                    np.moveaxis(image, 0, -1),
                    # move channels to last dimension
                    cmap=args.cmap)
                # plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
                # plt.axis('off')
                if args.plot_index % cols == 1:
                    plt.ylabel(ylabel_text)

            return predicted_label, confidence, L2_distance

        # Original image.
        original_label, original_confidence, original_L2_distance = show_image(
            image=original_image,
            original_image=original_image,
            title="Original")

        image = original_image

        # from_file = False
        file_name = args.dataset + "-roundedfft" + "-vals-per-channel-" + str(
            args.values_per_channel) + "-img-idx-" + str(
            args.index) + "-graph-recover"
        # "-compress-fft-layer-" + str(args.compress_fft_layer) +
        full_name = file_name + ".npy"
        adversarial_timing = "N/A"
        adversarial = None

        if args.is_adv_attack and attack.name() == "CarliniWagnerL2Attack":
            if os.path.exists(full_name):
                adversarial = np.load(file=full_name)
            else:
                start_adv = time.time()
                adversarial = attack(original_image, args.original_class_id)
                adversarial_timing = time.time() - start_adv
            image = adversarial

        # The rounded image.
        rounded_label = "N/A"
        rounded_confidence = "N/A"
        rounded_L2_distance = "N/A"
        if args.values_per_channel > 0 and image is not None:
            rounder = DenormRoundNorm(
                mean_array=args.mean_array, std_array=args.std_array,
                values_per_channel=args.values_per_channel)
            rounded_image = rounder.round(image)
            image = rounded_image
            # rounder = RoundingTransformation(
            #     values_per_channel=args.values_per_channel,
            #     rounder=np.around)
            # rounded_image = rounder(image)
            print("rounded_image min and max: ", rounded_image.min(), ",",
                  rounded_image.max())
            rounded_label, rounded_confidence, rounded_L2_distance = show_image(
                image=rounded_image,
                original_image=original_image,
                title="Rounded")
            print("show diff between input image and rounded: ",
                  np.sum(np.abs(rounded_image - original_image)))

        fft_label = "N/A"
        fft_confidence = "N/A"
        fft_L2_distance = "N/A"
        if args.compress_fft_layer > 0 and image is not None:
            compress_image = attack_round_fft.fft_complex_compression(
                image=image)
            image = compress_image
            title = "FFT Compressed: " + str(
                args.compress_fft_layer) + "%" + "\n"
            if args.interpolate == None:
                args.interpolate = "const"
            title += "interpolation: " + args.interpolate
            fft_label, fft_confidence, fft_L2_distance = show_image(
                image=compress_image,
                original_image=original_image,
                title=title)

        if args.is_adv_attack and attack.name() == "CarliniWagnerL2AttackRoundFFT":
            if os.path.exists(full_name):
                adversarial = np.load(file=full_name)
            else:
                start_adv = time.time()
                adversarial = attack(original_image, args.original_class_id)
                adversarial_timing = time.time() - start_adv

        if adversarial is None:
            # If the attack fails, adversarial will be None and a warning will
            # be printed.
            #     raise Exception(
            #         f"No adversarial was found for attack: {attack.name()},"
            #         f"and image index: {args.index}")
            #     print("adversarial image min and max: ", adversarial.min(), ",",
            #       adversarial.max())
            adversarial_label = "N/A"
            adversarial_confidence = "N/A"
            adversarial_L2_distance = "N/A"
            adversarial_timing = "N/A"
            # args.plot_index += 1  # do not show the image for None
        else:
            np.save(file=file_name, arr=adversarial)
            # adversarial = rounder.round(adversarial)
            adversarial_label, adversarial_confidence, adversarial_L2_distance = show_image(
                image=adversarial,
                original_image=original_image,
                title="Adversarial")

        # Show differences.
        # plt.title('Difference original vs. adversarial')
        # difference_adv_img = np.abs(adversarial - original_image)
        # print("max difference before round: ", np.max(difference_adv_img))
        # adversarial_normalized = denormalizer.denormalize(adversarial)
        # image_normalized = denormalizer.denormalize(image)
        # difference_adv_org = np.abs(
        #     adversarial_normalized - image_normalized)
        # print("pixel difference before round: ")
        # print("max: ", np.max(difference_adv_org))
        # print("sum: ", np.sum(difference_adv_org))

        # print("adversarial: ", adversarial)
        # adversarial = np.round(adversarial * 255) / 255
        # difference = np.abs(adversarial * 255 - image * 255)
        # print("max pixel difference after round: ", np.max(difference))
        # print("difference:\n", difference)
        # rounded_image_normalized = denormalizer.denormalize(rounded_image)
        # difference_adv_round = np.abs(
        #     adversarial_normalized - rounded_image_normalized)
        # print("pixel difference between rounded and adversarial images ")
        # print("max: ", np.max(difference_adv_round))
        # print("sum: ", np.sum(difference_adv_round))
        # print("difference:\n", difference)
        # https://www.statisticshowto.datasciencecentral.com/normalized/
        # difference = (difference - difference.min()) / (
        #         difference.max() - difference.min())
        # plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)

        # plt.imshow(np.moveaxis(difference, 0, -1))

        # print 0th channel only
        # print_color_map(difference[0], fig=fig, ax=ax)

        # print_color_map(difference_adv_org.sum(axis=0), fig=fig, ax=ax)

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # ax = plt.subplot(rows, cols, args.plot_index)
        # plt.title('Difference rounded vs. adversarial')
        # args.plot_index += 1
        # im = ax.imshow(difference_adv_round.sum(axis=0))
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax = cax)

        # plt.imshow(difference_adv_round.sum(axis=0), cmap='hot',
        #            interpolation='nearest')
        # plt.axis('off')

        # heat_map = difference_adv_round.sum(axis=0)
        # print_heat_map(input_map=heat_map,
        #                title="Difference rounded vs. adversarial")

        # interpolate = "log"
        # compress_image = FFTBandFunction2D.forward(
        #     ctx=None,
        #     input=torch.from_numpy(adversarial).unsqueeze(0),
        #     compress_rate=compress_rate).numpy().squeeze()

        # Write labels to the file.
        with open(args.file_name_labels, "a") as f:
            f.write(
                ";".join([str(x) for x in [
                    args.index,
                    args.original_label,  # ImageNet label (ground truth)
                    original_label,  # for the full model
                    rounded_label,
                    fft_label,
                    adversarial_label,
                    original_confidence,
                    rounded_confidence,
                    fft_confidence,
                    adversarial_confidence,
                    original_L2_distance,
                    rounded_L2_distance,
                    fft_L2_distance,
                    adversarial_L2_distance,
                    adversarial_timing
                ]]) + "\n")

        def print_fft(image, channel, title="", args=args):
            print("fft: ", title)
            print("input min: ", image.min())
            print("input max: ", image.max())
            xfft = to_fft(image, fft_type=fft_type)
            xfft = xfft[channel, ...]
            # torch.set_printoptions(profile='full')
            # print("original_fft size: ", original_fft.shape)
            # options = np.get_printoptions()
            # np.set_printoptions(threshold=np.inf)
            # torch.set_printoptions(profile='default')

            # save to file
            if args.save_out:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                output_path = os.path.join(dir_path,
                                           get_log_time() + title + "_fft.csv")
                print("output path: ", output_path)
                np.save(output_path, xfft)

            # go back to the original print size
            # np.set_printoptions(threshold=options['threshold'])
            xfft = xfft[:lim_y, :lim_x]
            # print("original_fft:\n", original_fft)
            # print_color_map(xfft, fig, ax)
            ylabel = "fft-ed channel " + str(channel) + ":\n" + fft_type
            print_heat_map(xfft, args=args, title=title, ylabel=ylabel)

            return xfft

        for fft_type in fft_types:
            for channel in channels:
                image_fft = print_fft(image=original_image, channel=channel,
                                      title="Original")

                if args.values_per_channel > 0:
                    rounded_fft = print_fft(image=rounded_image,
                                            channel=channel,
                                            title="Rounded")

                if args.compress_fft_layer > 0:
                    compressed_fft = print_fft(image=compress_image,
                                               channel=channel,
                                               title="FFT compressed")

                if adversarial is not None:
                    adversarial_fft = print_fft(image=adversarial,
                                                channel=channel,
                                                title="Adversarial")

        plt.subplots_adjust(hspace=0.6)

    format = 'png'  # "pdf" or "png" file_name
    file_name = "images/" + attack.name() + "-round-fft-" + str(
        args.compress_fft_layer) + "-" + args.dataset + "-channel-" + str(
        channels_nr) + "-" + "val-per-channel-" + str(
        args.values_per_channel) + "-" + "img-idx-" + str(
        args.index) + "-" + get_log_time()
    print("file name: ", file_name)
    plt.savefig(fname=file_name + "." + format, format=format)
    # plt.show(block=True)
    plt.close()
    return original_label, rounded_label, fft_label, adversarial_label


def index_ranges(
        input_ranges=[(20, 58), (2500, 2516), (5000, 5050), (9967, 10000)]):
    """
    Generate list of indexes with the given pair of inclusive ranges.

    :param ranges:
    :return:
    """
    indexes = np.array([], dtype=np.int)
    for range_start, range_end in input_ranges:
        indexes = np.concatenate(
            (indexes, [x for x in range(range_start, range_end + 1)]))
    return indexes


def result_file(args):
    args.file_name_labels = args.interpolate + "-round-fft-" + str(
        args.compress_fft_layer) + "-" + args.dataset + "-" + "val-per-channel-" + str(
        args.values_per_channel) + "-" + get_log_time()
    with open(args.file_name_labels, "a") as f:
        f.write(args.get_str() + "\n\n")
        f.write(";".join(["index",
                          args.dataset + " original label",
                          "full model label",
                          "rounded label",
                          "fft compressed label",
                          "adversarial label",
                          "original_confidence",
                          "rounded_confidence",
                          "fft_confidence",
                          "adversarial_confidence",
                          "original_L2_distance",
                          "rounded_L2_distance",
                          "fft_L2_distance",
                          "adversarial_L2_distance",
                          "adversarial timing",
                          ]) + "\n")


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    # save fft representations of the original and adversarial images to files
    args.save_out = False
    # args.diff_type = "source"  # "source" or "fft"
    args.diff_type = "fft"
    # args.dataset = "cifar10"  # "cifar10" or "imagenet"
    # args.dataset = "imagenet"
    # args.dataset = "mnist"
    # args.index = 13  # index of the image (out of 20) to be used
    # args.compress_rate = 0
    args.interpolate = "exp"
    args.use_foolbox_data = True
    args.is_adv_attack = True

    # args.compress_fft_layer = 5
    # args.values_per_channel = 8

    if torch.cuda.is_available() and args.use_cuda:
        print("cuda is available")
        args.device = torch.device("cuda")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("cuda id not available")
        args.device = torch.device("cpu")
    # for values_per_channel in [2**x for x in range(1,8,1)]:
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in range(2, 256, 1):
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in [2]:
    #     args.index = 1
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in [8]:
    #     for index in range(0, 17):  # 12, 13, 16
    #         args.index = index
    #         args.values_per_channel = values_per_channel
    #         run(args)
    # for values_per_channel in [8]:
    #     args.index = 13
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in [8]:
    #     args.index = 16
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in [8]:
    #     args.values_per_channel = values_per_channel
    #     for index in range(20):
    #         args.index = index
    #         start = time.time()
    #         run(args)
    #         print("elapsed time: ", time.time() - start)
    # for index in range(20):
    #     args.index = index
    #     for values_per_channel in [2**x for x in range(1,8,1)]:
    #         args.values_per_channel = values_per_channel
    #         run(args)
    # for interpolate in ["exp", "log", "const", "linear"]:
    # for interpolate in ["exp"]:
    #     args.interpolate = interpolate
    #     result_file(args)
    #     for values_per_channel in [0]:
    #         args.values_per_channel = values_per_channel
    #         # indexes = index_ranges([(0, 49999)])  # all validation ImageNet
    #         # print("indexes: ", indexes)
    #         for index in range(args.start_epoch, 10000):
    #             args.index = index
    #             print(args.get_str())
    #             start = time.time()
    #             run(args)
    #             print("single run elapsed time: ", time.time() - start)

    print(args.get_str())
    # for interpolate in ["exp"]:
    #     args.interpolate = interpolate
    out_fft_recovered_file = "out_fft_recovered" + str(
        args.dataset) + "-" + str(args.values_per_channel) + ".txt"
    with open(out_fft_recovered_file, "a") as f:
        f.write("compress_fft_layer,"
                "% or recovered,"
                "# of recovered\n")
    args.interpolate = "exp"
    # for compress_fft_layer in [1, 2, 3, 5, 10, 15, 25, 35, 50, 60, 75, 80, 90]:
    for compress_fft_layer in range(1, 100):
        print("compress_fft_layer: ", compress_fft_layer)
        args.compress_fft_layer = compress_fft_layer
        result_file(args)
        # indexes = index_ranges([(0, 49999)])  # all validation ImageNet
        # print("indexes: ", indexes)
        count_recovered_fft = 0
        total_count = 0
        for index in range(args.start_epoch, args.sample_count_limit):
            total_count += 1
            args.index = index
            print("image index: ", index)
            start = time.time()
            original_label, rounded_label, fft_label, adversarial_label = run(
                args)
            if original_label == fft_label:
                count_recovered_fft += 1
            print("single run elapsed time: ", time.time() - start)
        with open(out_fft_recovered_file, "a") as f:
            f.write(",".join([str(x) for x in
                              [compress_fft_layer,
                               count_recovered_fft / total_count * 100,
                               count_recovered_fft]]) + "\n")

    print("total elapsed time: ", time.time() - start_time)
