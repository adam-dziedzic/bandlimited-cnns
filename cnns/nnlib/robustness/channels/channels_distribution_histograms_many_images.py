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
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.load_data import get_data
from cnns.nnlib.utils.exec_args import get_args
import time

beg = time.time()

cur_dir = os.path.dirname(os.path.realpath(__file__))

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)
MY_GOLD = (148, 139, 61)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_GOLD]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = [":", "-", "--", ":", "-", "--", ":", "-"]
legend_size = 22
line_width = 2
markersize = 20

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

args = get_args()
if args.is_debug:
    args.use_foolbox_data = False
if not args.use_foolbox_data:
    train_loader, test_loader, train_dataset, test_dataset, limit = get_data(
        args=args)

# string out the params for channels
func = 'func'
name = 'name'
value = 'value'
deltas = 'deltas'
l2_dist = 'l2_dist'
graph_label = 'graph_label'
single_graph = True

unif = {
    func: unif,
    name: 'unif',
    value: 0.03,
    graph_label: 'unif: $\epsilon=0.03$'
}

gauss = {
    func: gauss,
    name: 'gauss',
    value: 0.03,
    graph_label: 'gauss: $\sigma=0.03$'
}

laplace = {
    func: laplace,
    name: 'laplace',
    value: 0.03,
    graph_label: 'laplace: $\mu=0$, $b=0.03$'
}

fft = {
    func: fft,
    name: 'fft',
    value: 20,
    graph_label: 'fft: 20% compression'
}

round = {
    func: round,
    name: 'cd',
    value: 32,
    graph_label: 'cd: 5 bits'
}

svd = {
    func: svd,
    name: 'svd',
    value: 50,
    graph_label: 'svd: 50% compression'
}

sub = {
    func: sub,
    name: 'RGB subtract',
    value: 10,
    graph_label: 'RGB subtract: value 10'
}

channels = [
    # unif,
    gauss,
    laplace,
    fft,
    round,
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
    if index > 1:
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
    full_rel_path = 'cache/adversarial_image' + params_str + '.npy'
    file_name = os.path.join(cur_dir, full_rel_path)
    print('file_name: ', file_name)
    if os.path.exists(file_name):
        adversarial_image = np.load(file_name)
    else:
        adversarial_image = attack(image, label, max_iterations=max_iters,
                                   initial_const=init_const,
                                   confidence=confidence)
    if not os.path.exists(file_name):
        np.save(file=file_name, arr=adversarial_image)

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
        l2_distance = np.sqrt(np.sum(delta * delta))
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
    else:
        plt.subplot(nrows, ncols, 1)

    data = {}
    for param in [deltas, l2_dist]:
        data[param] = np.average(channel[param], axis=0)

    if single_graph:
        hist_label = channel[graph_label]
    else:
        hist_label = ''
    print('hist label: ', hist_label)

    if not single_graph:
        n, bins, patches = plt.hist(
            x=data[deltas],
            bins="auto",
            color=colors[i],
            linestyle=linestyles[i],
            lw=line_width,
            alpha=0.7,
            rwidth=0.85,
            histtype='step',
            label=hist_label
        )
        plt.grid(axis="y", alpha=0.75)
        # plt.legend(loc='upper right',
        #            ncol=1,
        #            # frameon=True,
        #            prop={'size': legend_size},
        #            # bbox_to_anchor=dataset[bbox]
        #            )
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        channel_info = channel[name] + ' L2 dist: ' + f'{data[l2_dist]:9.4f}'
        plt.title(channel_info)
    else:
        y, bin_edges = np.histogram(data[deltas], bins=1000)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        # data = np.array(np.random.rand(1000))
        # y, binEdges = np.histogram(data, bins=100)
        # bin_centers = 0.5 * (binEdges[1:] + binEdges[:-1])
        # plt.plot(bin_centers, y,
        #          color=colors[i % len(colors)],
        #          linestyle=linestyles[i % len(linestyles)],
        #          label=hist_label,
        #          lw=line_width,
        #          marker=markers[i % len(markers)],
        #          markersize=markersize,
        #          )

        plt.plot(
            bin_centers,
            y,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            lw=line_width,
            # marker=markers[i % len(markers)],
            # markersize=markersize,
            label=hist_label
        )

    # plt.text(23, 45, r"$\mu=15, b=3$")
    # maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(
    #     ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

if single_graph:
    plt.legend(loc='upper left',
               ncol=1,
               frameon=True,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )
    plt.grid()
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.subplots_adjust(hspace=1.5)
plt.subplots_adjust(wspace=1.5)
plt.savefig('graphs/' + get_log_time() + 'channel_histogram5.pdf',
            bbox_inches='tight')
plt.show()

print('Works only for a single channel:')
print(f"\nBase test accuracy of the model: "
      f"{original_count / recover_count}")
print(f"\nAccuracy of the model after attack: "
      f"{adversarial_count / recover_count}")
print(f"\nAccuracy of the model after noising: "
      f"{defended_count / recover_count}")

print('total elapsed time: ', time.time() - beg)
