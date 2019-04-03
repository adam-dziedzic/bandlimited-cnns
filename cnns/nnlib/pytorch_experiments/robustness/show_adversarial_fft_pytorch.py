#  Band-limiting
#  Copyright (c) 2019. Adam Dziedzic
#  Licensed under The Apache License [see LICENSE for details]
#  Written by Adam Dziedzic

# if you use Jupyter notebooks
# %matplotlib inline

from cnns.nnlib.utils.general_utils import get_log_time
import matplotlib.pyplot as plt
import foolbox
import numpy as np
import torch
import torchvision.models as models
# from cnns.nnlib.pytorch_layers.pytorch_utils import get_full_energy
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from cnns.nnlib.pytorch_layers.pytorch_utils import get_phase
import os
from cnns.nnlib.benchmarks.imagenet_from_class_idx_to_label import \
    imagenet_from_class_idx_to_label

# from scipy.special import softmax

# arguments
# save fft representations of the original and adversarial images to files
save_out = False
diff_type = "source"  # "source" or "fft"


def softmax(x):
    s = np.exp(x - np.max(x))
    s /= np.sum(s)
    return s


def to_fft(x, is_log=True):
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
    if is_log:
        return 20 * np.log10(xfft.numpy())
    else:
        return xfft.numpy()


def to_fft_phase(xfft):
    # The phase is unwrapped using the unwrap function so that we can see a
    # continuous function of frequency.
    return np.unwrap(get_phase(xfft).numpy())


def znormalize(x):
    return (x - x.min()) / (x.max() - x.min())


init_y, init_x = 224, 224
lim_y, lim_x = init_y, init_x
# lim_y, lim_x = init_y // 2, init_x // 2
# lim_y, lim_x = 2, 2
# lim_y, lim_x = 3, 3
images, labels = foolbox.utils.samples(dataset='imagenet', index=0,
                                       batchsize=20, shape=(init_y, init_x),
                                       data_format='channels_first')

images = images / 255
resnet = models.resnet50(
    pretrained=True).cuda().eval()  # for CPU, remove cuda()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(resnet, bounds=(0, 1),
                                     num_classes=1000,
                                     preprocessing=(mean, std))

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


# choose how many channels should be plotted
channels_nr = 1
channels = [x for x in range(channels_nr)]
# attacks = [foolbox.attacks.FGSM(fmodel),
#            foolbox.attacks.AdditiveUniformNoiseAttack(fmodel)]
attacks = [foolbox.attacks.FGSM(fmodel)]
rows = len(attacks) * (1 + 2 * len(channels))
cols = 3

fig = plt.figure(figsize=(30, 30))

# index for each subplot
i = 1

for attack in attacks:
    # get source image and label, idx - is the index of the image
    # image, label = foolbox.utils.imagenet_example()
    idx = 0
    image, label = images[idx], labels[idx]

    print("original label: id:", label, ", class: ",
          imagenet_from_class_idx_to_label[label])

    predictions_original, _ = fmodel.predictions_and_gradient(image=image,
                                                              label=label)
    # predictions_original = znormalize(predictions_original)
    predictions_original = softmax(predictions_original)
    original_prediction = imagenet_from_class_idx_to_label[
        np.argmax(predictions_original)]
    print("model original prediction: ", original_prediction)
    original_confidence = np.max(predictions_original)
    sum_predictions = np.sum(predictions_original)
    # print("sum predictions: ", sum_predictions)
    # print("predictions_original: ", predictions_original)

    plt.subplot(rows, cols, i)
    i += 1
    plt.title('Original\nlabel: ' + str(
        original_prediction.replace(",",
                                    "\n") + "\nconfidence: " + str(
            np.around(original_confidence, decimals=decimals))))
    plt.imshow(np.moveaxis(image, 0, -1))  # move channels to last dimension
    # plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(rows, cols, i)
    i += 1

    adversarial = attack(image, label)
    predictions_adversarial, _ = fmodel.predictions_and_gradient(
        image=adversarial, label=label)
    # predictions_adversarial = znormalize(predictions_adversarial)
    predictions_adversarial = softmax(predictions_adversarial)
    adversarial_prediction = imagenet_from_class_idx_to_label[
        np.argmax(predictions_adversarial)]
    adversarial_confidence = np.max(predictions_adversarial)

    print("model adversarial prediciton: ", adversarial_prediction)

    # if the attack fails, adversarial will be None and a warning will be printed
    if adversarial is None:
        raise Exception('foolbox did not find an adversarial example')

    plt.title('Adversarial ' + attack.name() + '\nlabel: ' + str(
        adversarial_prediction.replace(",",
                                       "\n") + "\nconfidence: " + str(
            np.around(adversarial_confidence, decimals=decimals))))
    plt.imshow(np.moveaxis(adversarial, 0, -1))
    plt.axis('off')

    plt.subplot(rows, cols, i)
    i += 1
    plt.title('Difference')
    print("adversarial: ", adversarial)
    adversarial = np.round(adversarial * 255) / 255
    difference = np.abs(adversarial - image)
    print("max difference: ", np.max(difference) * 255)
    print("difference:\n", difference)
    # https://www.statisticshowto.datasciencecentral.com/normalized/
    # difference = (difference - difference.min()) / (
    #         difference.max() - difference.min())
    # plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.imshow(np.moveaxis(difference, 0, -1))
    plt.axis('off')

    for fft_type in ["magnitude", "phase"]:
        for channel in channels:
            ax = plt.subplot(rows, cols, i)
            i += 1
            # plt.title('Original\nfft-ed')
            original_fft = to_fft(image)
            original_fft = original_fft[channel, ...]
            # torch.set_printoptions(profile='full')
            # print("original_fft size: ", original_fft.shape)
            # options = np.get_printoptions()
            # np.set_printoptions(threshold=np.inf)
            # torch.set_printoptions(profile='default')

            # save to file
            if save_out:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                output_path = os.path.join(dir_path, "original_fft.csv")
                print("output path: ", output_path)
                np.save(output_path, original_fft)

            # go back to the original print size
            # np.set_printoptions(threshold=options['threshold'])
            original_fft = original_fft[:lim_y, :lim_x]
            # print("original_fft:\n", original_fft)
            print_color_map(original_fft, fig, ax)

            # plt.axis('off')
            plt.ylabel("fft-ed channel " + str(channel) + ":\n" + fft_type)

            ax = plt.subplot(rows, cols, i)
            i += 1
            # plt.title('Adversarial\nfft-ed')
            adversarial_fft = to_fft(adversarial)
            adversarial_fft = adversarial_fft[channel]

            if save_out:
                output_path = os.path.join(dir_path, "adversarial_fft.csv")
                print("output path: ", output_path)
                np.save(output_path, adversarial_fft)

            adversarial_fft = adversarial_fft[:lim_y, :lim_x]
            # print("adversarial fft:\n", adversarial_fft)

            print_color_map(adversarial_fft, fig, ax)

            # plt.axis('off')

            # plt.subplot(2, 3, 6)
            # plt.title('FFT Difference')
            # difference_fft = adversarial_fft - original_fft
            # final_difference = difference_fft / abs(difference_fft).max() * 0.2 + 0.5
            # plt.imshow(final_difference)
            # heatmap_legend = plt.pcolor(final_difference)
            # plt.colorbar(heatmap_legend)
            # plt.axis('off')

            ax = plt.subplot(rows, cols, i)
            i += 1
            # plt.title('Difference\nfft-ed')

            if diff_type == "source":
                difference = np.abs((image / 255) - (adversarial / 255))
                difference_fft = to_fft(difference, is_log=False)
                difference_fft = difference_fft[channel]
                difference_fft = difference_fft[:lim_y, :lim_x]
            elif diff_type == "fft":
                # difference = (adversarial_fft - original_fft)[..., np.newaxis]
                # difference_fft = to_fft(difference)
                difference_fft = adversarial_fft - original_fft
            else:
                raise Exception(f"Unknown diff_type: {diff_type}")

            print_color_map(difference_fft, fig, ax)
            # plt.axis('off')

    plt.subplots_adjust(hspace=0.6)

format = 'pdf'
file_name = "images/" + attack.name() + "-channel-" + str(
    channel) + "-" + get_log_time()
plt.savefig(fname=file_name + "." + format, format=format)
plt.show(block=True)
plt.close()
