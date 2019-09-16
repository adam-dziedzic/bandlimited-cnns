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
import os
import torch
import torchvision.models as models
import numpy as np
import foolbox
# %pylab inline
from cnns.nnlib.robustness.utils import show_image
from cnns.nnlib.robustness.channels.histogram_matplotlib import \
    plot_hist
from matplotlib import pyplot as plt
from cnns.nnlib.robustness.channels.channels_definition import \
    uniform_noise_numpy as unif
from cnns.nnlib.robustness.channels.channels_definition import \
    gauss_noise_numpy as gauss
from cnns.nnlib.robustness.channels.channels_definition import \
    laplace_noise_numpy_subtract as laplace
from cnns.nnlib.robustness.channels.channels_definition import \
    logistic_noise_numpy as logistic
from cnns.nnlib.robustness.channels.channels_definition import \
    round_numpy as round
from cnns.nnlib.robustness.channels.channels_definition import \
    fft_numpy as fft
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_numpy as svd
from cnns.nnlib.robustness.channels.channels_definition import \
    subtract_rgb_numpy as sub
from cnns.nnlib.utils.general_utils import get_log_time


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

# string out the params for channels
func = 'func'
name = 'name'
value = 'value'
deltas = 'deltas'
l2_dist = 'l2_dist'

single_graph = True

unif = {
    func: unif,
    name: 'unif',
    value: 0.03,
}

gauss = {
    func: gauss,
    name: 'gauss',
    value: 0.03,
}

laplace = {
    func: laplace,
    name: 'laplace',
    value: 0.003,
}

fft = {
    func: fft,
    name: 'fft',
    value: 50,
}

round = {
    func: round,
    name: 'cd',
    value: 16,
}

svd = {
    func: svd,
    name: 'svd',
    value: 50,
}

sub = {
    func: sub,
    name: 'RGB subtract',
    value: 10,
}

channels = [
    # unif,
    # gauss,
    laplace,
    # fft,
    # round,
    svd,
    # sub,
]

# We collect data for each channel.
for channel in channels:
    channel[deltas] = []
    channel[l2_dist] = []

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

    max_iters = 1000
    init_const = 0.01
    confidence = 0
    params = [index, max_iters, init_const, confidence]
    params_str = '_'.join([str(x) for x in params])
    file_name = 'adversarial_image' + params_str + '.npy'
    print('file_name: ', file_name)
    if os.path.isfile(file_name):
        adversarial_image = np.load(file_name)
    else:
        adversarial_image = attack(image, label, max_iterations=max_iters,
                                   initial_const=init_const,
                                   confidence=confidence)
    if not os.path.isfile(file_name):
        np.save(file=file_name + ".npy", arr=adversarial_image)

    print('adversarial image min, max:', adversarial_image.min(),
          adversarial_image.max())

    adversarial_predictions = model.predictions(adversarial_image)
    adversarial_prediciton = np.argmax(adversarial_predictions)
    print("adversarial prediction: ", adversarial_prediciton)
    if adversarial_prediciton == label:
        adversarial_count += 1

    for i, channel in enumerate(channels):
        noise_func = channel[func]
        noise_name = channel[name]
        noise_value = channel[value]

        print('noise name: ', noise_name)
        noised_image = noise_func(adversarial_image, noise_value)
        noised_image = noised_image.astype(np.float32)

        noise_predictions = model.predictions(noised_image)
        noise_prediction = np.argmax(noise_predictions)
        print("noise prediction: ", noise_prediction)
        if noise_prediction == label:
            defended_count += 1

        delta = noised_image - image
        l2_distance = np.sqrt(np.sum(deltas * deltas))
        delta = delta.flatten()
        channel[deltas].append(delta)
        channel[l2_dist].append(l2_distance)
        # plot_hist(delta)

if single_graph:
    ncols = 1
    nrows = 1
else:
    ncols = 1 if len(channels) == 1 else 2
    nrows = max(len(channels) // 2, 1)
    plt.subplots(nrows=nrows, ncols=ncols)

for i, channel in enumerate(channels):
    if not single_graph:
        plt.subplot(nrows, ncols, i + 1)

    data = {}
    for param in [deltas, l2_dist]:
        data[param] = np.average(channel[param], axis=0)

    if single_graph:
        label = channel[name] + ' ' + str(channel[value])
    else:
        label = ''

    n, bins, patches = plt.hist(
        x=data[deltas],
        bins="auto",
        color="#0504aa",
        alpha=0.7,
        rwidth=0.85,
        histtype='step',
        label=label,
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if not single_graph:
        plt.title(channel[name] + ' L2 dist: ' + f'{data[l2_dist]:9.4f}')
    # plt.text(23, 45, r"$\mu=15, b=3$")
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(
        ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.subplots_adjust(hspace=1.5)
plt.subplots_adjust(wspace=1.5)
plt.savefig('graphs/' + get_log_time() + 'channel_histogram5.pdf',
            bbox_inches='tight')
plt.show()
