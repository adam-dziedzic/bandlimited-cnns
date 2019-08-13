"""
A simple example that shows that adding a uniform noise after the FGSM attack
can restore the correct label. The labels are given as numbers from 0 to 999.
We use ResNet-18 on 20 ImageNet samples from foolbox.

Install PyTorch 1.1: https://pytorch.org/

Install foolbox 1.9 (this version is necessary):
git clone https://github.com/bethgelab/foolbox.git
cd foolbox
git reset --hard 5191c3a595baadedf0a3659d88b48200024cd534
pip install --editable .
"""

import torch
import torchvision.models as models
import numpy as np
import foolbox
# %pylab inline
from cnns.nnlib.robustness.utils import show_image
from cnns.nnlib.robustness.channel_histograms.histogram_matplotlib import \
    plot_hist
from matplotlib import pyplot as plt
from cnns.nnlib.robustness.channel_histograms.channels_definition import \
    uniform_noise_numpy as unif
from cnns.nnlib.robustness.channel_histograms.channels_definition import \
    gauss_noise_numpy as gauss
from cnns.nnlib.robustness.channel_histograms.channels_definition import \
    laplace_noise_numpy as laplace
from cnns.nnlib.robustness.channel_histograms.channels_definition import \
    logistic_noise_numpy as logistic
from cnns.nnlib.robustness.channel_histograms.channels_definition import \
    round_numpy as round
from cnns.nnlib.robustness.channel_histograms.channels_definition import \
    fft_numpy as fft
from cnns.nnlib.robustness.channel_histograms.channels_definition import \
    compress_svd_numpy as svd
from cnns.nnlib.robustness.channel_histograms.channels_definition import \
    subtract_rgb_numpy as sub

# Settings for the PyTorch model.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

imagenet_mean_array = np.array(imagenet_mean, dtype=np.float32).reshape(
    (3, 1, 1))
imagenet_std_array = np.array(imagenet_std, dtype=np.float32).reshape(
    (3, 1, 1))

# The min/max value per pixel after normalization.
imagenet_min = np.float32(-2.1179039478302)
imagenet_max = np.float32(2.640000104904175)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# Instantiate the model.
resnet = models.resnet50(pretrained=True)
resnet.to(device)
resnet.eval()

model = foolbox.models.PyTorchModel(resnet, bounds=(0, 1), num_classes=1000,
                                    preprocessing=(imagenet_mean_array,
                                                   imagenet_std_array))
original_count = 0
adversarial_count = 0
defended_count = 0
recover_count = 20

images, labels = foolbox.utils.samples("imagenet", data_format="channels_first",
                                       batchsize=recover_count)
images = images / 255  # map from [0,255] to [0,1] range

channels = [
    [round, 'cd', 16],
    [fft, 'fft', 50],
    [unif, 'unif', 0.03],
    [gauss, 'gauss', 0.03],
    [laplace, 'laplace', 0.03],
    [logistic, 'logistic', 0.03],
    [svd, 'svd', 50],
    [sub, 'RGB subtract', 10]
]

fontsize = 10
legend_size = 10
title_size = 10
font = {'size': fontsize}
import matplotlib

matplotlib.rc('font', **font)

width = 50
height = 100
fig = plt.figure(figsize=(width, height))

for index, (label, image) in enumerate(zip(labels, images)):
    if index != 11:
        continue
    print("\nimage index: ", index)

    print("true prediction: ", label)

    # show_image(image)

    # Original prediction of the model (without any adversarial changes or noise).
    original_predictions = model.predictions(image)
    original_prediction = np.argmax(original_predictions)
    print("original prediction: ", original_prediction)
    if original_prediction == label:
        original_count += 1

    # Attack the image.
    # attack = foolbox.attacks.FGSM(model)
    # attack = foolbox.attacks.L1BasicIterativeAttack(model)
    attack = foolbox.attacks.CarliniWagnerL2Attack(model)
    adversarial_image = attack(image, label, max_iterations=100)

    print('adversarial image min, max:', adversarial_image.min(),
          adversarial_image.max())

    adversarial_predictions = model.predictions(adversarial_image)
    adversarial_prediciton = np.argmax(adversarial_predictions)
    print("adversarial prediction: ", adversarial_prediciton)
    if adversarial_prediciton == label:
        adversarial_count += 1

    ncols = 1 if len(channels) == 1 else 2
    nrows = max(len(channels) // 2, 1)
    plt.subplots(nrows=nrows, ncols=ncols)
    for i, noiser in enumerate(channels):
        noise_func = noiser[0]
        noise_name = noiser[1]
        noise_value = noiser[2]

        print('noise name: ', noise_name)
        noised_image = noise_func(adversarial_image, noise_value)
        noised_image = noised_image.astype(np.float32)

        noise_predictions = model.predictions(noised_image)
        noise_prediction = np.argmax(noise_predictions)
        print("uniform noise prediction: ", noise_prediction)
        if noise_prediction == label:
            defended_count += 1

        deltas = noised_image - image
        l2_dist = np.sqrt(np.sum(deltas * deltas))
        deltas = deltas.flatten()
        # plot_hist(deltas)
        plt.subplot(nrows, ncols, i + 1)
        n, bins, patches = plt.hist(
            x=deltas, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
        )
        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(noise_name + ' L2 dist: ' + f'{l2_dist:9.4f}')
        # plt.text(23, 45, r"$\mu=15, b=3$")
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(
            ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    plt.subplots_adjust(hspace=1.5)
    plt.subplots_adjust(wspace=1.5)
    plt.savefig(noise_name + '_channel_histogram.pdf', bbox_inches='tight')
    plt.show()

print(f"\nBase test accuracy of the model: "
      f"{original_count / recover_count}")
print(f"\nAccuracy of the model after attack: "
      f"{adversarial_count / recover_count}")
print(f"\nAccuracy of the model after noising: "
      f"{defended_count / recover_count}")
