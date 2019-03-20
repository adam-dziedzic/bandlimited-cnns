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
from cnns.nnlib.pytorch_layers.pytorch_utils import get_full_energy
import os
from cnns.nnlib.benchmarks.imagenet_from_class_idx_to_label import \
    imagenet_from_class_idx_to_label
# from scipy.special import softmax

def softmax(x):
    s = np.exp(x - np.max(x))
    s /= np.sum(s)
    return s

def to_fft(x):
    # x = torch.from_numpy(x)
    x = torch.tensor(x)
    x = x.permute(2, 0, 1)  # move channel as the first dimension
    xfft = torch.rfft(x, onesided=False, signal_ndim=2)
    _, xfft_squared = get_full_energy(xfft)
    xfft_abs = torch.sqrt(xfft_squared)
    # xfft_abs = xfft_abs.sum(dim=0)
    return np.log(xfft_abs.numpy())


def znormalize(x):
    return (x - x.min()) / (x.max() - x.min())


init_y, init_x = 224, 224
images, labels = foolbox.utils.samples(dataset='imagenet', index=0,
                                       batchsize=20, shape=(init_y, init_x),
                                       data_format='channels_first')

# instantiate model
images = images / 255

# instantiate the model
resnet = models.resnet50(
    pretrained=True).cuda().eval()  # for CPU, remove cuda()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(resnet, bounds=(0, 1),
                                     num_classes=1000,
                                     preprocessing=(mean, std))
# setting for the heat map
cmap = 'hot'
interpolation = 'nearest'

# get source image and label
# image, label = foolbox.utils.imagenet_example()
idx = 0
image, label = images[idx], labels[idx]

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB

attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image, label)

print("original label: id:", label, ", class: ",
      imagenet_from_class_idx_to_label[label])

# predictions_original = kmodel.predict(image[np.newaxis, :, :, ::-1])
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

# predictions_adversarial = kmodel.predict(adversarial[np.newaxis, ...])
predictions_adversarial, _ = fmodel.predictions_and_gradient(
    image=adversarial, label=label)
# predictions_adversarial = znormalize(predictions_adversarial)
predictions_adversarial = softmax(predictions_adversarial)
adversarial_prediction = imagenet_from_class_idx_to_label[
    np.argmax(predictions_adversarial)]
adversarial_confidence = np.max(predictions_adversarial)

print("model adversarial prediciton: ", adversarial_prediction)

# attack = foolbox.attacks.MultiplePixelsAttack(fmodel)
# adversarial = attack(image, label, num_pixels=100)

# attack = foolbox.attacks.AdditiveUniformNoiseAttack(fmodel)
# adversarial = attack(image[:, :, ::-1], label, epsilons=[0.4])

# adversarial = adversarial[:, :, ::-1].copy()  # from BGR to RGB
# print("adversarial: ", adversarial)
# if the attack fails, adversarial will be None and a warning will be printed

if adversarial is None:
    raise Exception('foolbox did not find an adversarial example')

channels = [x for x in range(3)]
rows = 1 + len(channels)
cols = 3

fig = plt.figure(figsize=(10, 15))
plt.subplot(rows, cols, 1)
plt.title('Original\nlabel: ' + str(
    original_prediction.replace(",",
                                "\n") + "\nconfidence: " + str(
        np.around(original_confidence, decimals=2))))
image = np.moveaxis(image, 0, -1)
plt.imshow(image)  # move channels to last dimension
# plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')

plt.subplot(rows, cols, 2)
plt.title('Adversarial\nlabel: ' + str(
    adversarial_prediction.replace(",",
                                   "\n") + "\nconfidence: " + str(
        np.around(adversarial_confidence, decimals=2))))
adversarial = np.moveaxis(adversarial, 0, -1)
plt.imshow(adversarial)
plt.axis('off')

plt.subplot(rows, cols, 3)
plt.title('Difference')
difference = adversarial - image
# https://www.statisticshowto.datasciencecentral.com/normalized/
difference = (difference - difference.min()) / (
        difference.max() - difference.min())
# plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.imshow(difference)
plt.axis('off')

i = 4
for channel in channels:
    lim_y, lim_x = init_y, init_x
    plt.subplot(rows, cols, i)
    i += 1
    # plt.title('Original\nfft-ed')
    original_fft = to_fft(image)
    original_fft = original_fft[channel]
    # torch.set_printoptions(profile='full')
    # print("original_fft size: ", original_fft.shape)
    # options = np.get_printoptions()
    # np.set_printoptions(threshold=np.inf)
    # torch.set_printoptions(profile='default')
    # print("original_fft: ", original_fft)

    # save to file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(dir_path, "original_fft.csv")
    print("output path: ", output_path)
    np.save(output_path, original_fft)

    # go back to the original print size
    # np.set_printoptions(threshold=options['threshold'])
    original_fft = original_fft[:lim_y, :lim_x]
    plt.imshow(original_fft, cmap=cmap,
               interpolation=interpolation)
    heatmap_legend = plt.pcolor(original_fft)
    plt.colorbar(heatmap_legend)
    # plt.axis('off')
    plt.ylabel("fft-ed\nchannel " + str(channel))

    plt.subplot(rows, cols, i)
    i += 1
    # plt.title('Adversarial\nfft-ed')
    adversarial_fft = to_fft(adversarial)
    adversarial_fft = adversarial_fft[channel]

    output_path = os.path.join(dir_path, "adversarial_fft.csv")
    print("output path: ", output_path)
    np.save(output_path, adversarial_fft)

    adversarial_fft = adversarial_fft[:lim_y, :lim_x]
    plt.imshow(adversarial_fft, cmap=cmap,
               interpolation=interpolation)
    heatmap_legend = plt.pcolor(adversarial_fft)
    plt.colorbar(heatmap_legend)
    # plt.axis('off')

    # plt.subplot(2, 3, 6)
    # plt.title('FFT Difference')
    # difference_fft = adversarial_fft - original_fft
    # final_difference = difference_fft / abs(difference_fft).max() * 0.2 + 0.5
    # plt.imshow(final_difference)
    # heatmap_legend = plt.pcolor(final_difference)
    # plt.colorbar(heatmap_legend)
    # plt.axis('off')

    plt.subplot(rows, cols, i)
    i += 1
    # plt.title('Difference\nfft-ed')
    # difference = np.abs((image / 255) - (adversarial / 255))
    difference = image - adversarial
    difference_fft = to_fft(difference)
    difference_fft = difference_fft[channel]
    difference_fft = difference_fft[:lim_y, :lim_x]
    plt.imshow(difference_fft, cmap=cmap,
               interpolation=interpolation)
    heatmap_legend = plt.pcolor(difference_fft)
    plt.colorbar(heatmap_legend)
    # plt.axis('off')

format = 'pdf'
file_name = "images/" + attack.name() + "-channel-" + str(
    channel) + "-" + get_log_time()
plt.savefig(fname=file_name + "." + format, format=format)
plt.show(block=True)
plt.close()
