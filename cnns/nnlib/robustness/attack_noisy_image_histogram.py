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
import matplotlib

print(matplotlib.get_backend())
# matplotlib.use('TkAgg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import numpy as np
import foolbox
from cnns.nnlib.robustness.utils import get_foolbox_model
# %pylab inline
from cnns.nnlib.robustness.utils import show_image
from cnns.nnlib.robustness.channels.histogram_matplotlib import \
    plot_hist
from cnns.nnlib.robustness.channels.channels_definition import \
    uniform_noise_numpy as unif
from cnns.nnlib.robustness.channels.channels_definition import \
    gauss_noise_numpy as gauss
from cnns.nnlib.robustness.channels.channels_definition import \
    laplace_noise_numpy as laplace
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

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


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

original_count = 0
adversarial_count = 0
defended_count = 0
recover_count = 20
bounds = (0, 1)

# Instantiate the model.
dataset = 'imagenet'

if dataset == 'imagenet':
    resnet = models.resnet50(pretrained=True)
    resnet.to(device)
    resnet.eval()

    model = foolbox.models.PyTorchModel(resnet, bounds=bounds, num_classes=1000,
                                        preprocessing=(imagenet_mean_array,
                                                       imagenet_std_array))
    images, labels = foolbox.utils.samples("imagenet",
                                           data_format="channels_first",
                                           batchsize=recover_count)
elif dataset == 'cifar10':
    model_path = 'saved_model_2019-05-16-11-37-45-415722-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.56-channel-vals-0.model'
    model = get_foolbox_model(args, model_path=model_path,
                              compress_rate=0,
                              min=cifar_min, max=cifar_max)

images = images / 255  # map from [0,255] to [0,1] range

fontsize = 25
legend_size = 25
title_size = 25
font = {'size': fontsize}
import matplotlib

matplotlib.rc('font', **font)

width = 50
height = 100
fig = plt.figure(figsize=(width, height))


# Attack the image.
class EmptyAttack(foolbox.attacks.Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        return input_or_adv


fgsm_attack = foolbox.attacks.FGSM(model)
bl1_attack = foolbox.attacks.L1BasicIterativeAttack(model)
cw_attack = foolbox.attacks.CarliniWagnerL2Attack(model)
pgd_attack = foolbox.attacks.RandomStartProjectedGradientDescentAttack(model)
empty_attack = EmptyAttack()

repeats = 100
noise_func = gauss
# noise_strengths = [0.0, 0.003, 0.1]
noise_strengths = [0.0, 0.001, 0.007, 0.009, 0.02, 0.04, 0.05, 0.08]
# noise_strengths = [0.0, 0.001, 0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08,
#                    0.09]
# noise_strengths = [0.0, 0.001, 0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08,
#                    0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# noise_strengths = []


# show_image(image)


# attacks = [fgsm_attack, bl1_attack]
# attacks = [cw_attack]
attacks = [fgsm_attack]
ncols = 1
nrows = max(1, len(attacks) // 2)
lw = 4
ncols_legend = 1
width = 15
height = 7
fig = plt.figure(figsize=(width * ncols, height * nrows))

colors = [get_color(color) for color in
          [MY_GREEN, MY_BLUE, MY_ORANGE, MY_RED, MY_BLACK, MY_RED, MY_GREEN]]
markers = ["+", "o", "v", "s", "D", "^", "+"]
linestyles = ["-", "-", "--", ":", ":", '--', '--']
markers = ['o', 'v', 's']

plt.subplot(nrows, ncols, 1)

for attack_iter, attack in enumerate(attacks):
    images = images[:1]
    labels = labels[:1]  # limit to a single image
    for image_idx, (image, label) in enumerate(zip(images, labels)):
        # Original prediction of the model (without any adversarial changes
        # or noise).
        original_predictions = model.predictions(image)
        original_prediction = np.argmax(original_predictions)
        print("original prediction: ", original_prediction)
        if original_prediction == label:
            original_count += 1

        distances = []
        org_class_counts = []
        adv_class_counts = []
        other_class_counts = []

        for noise_strength in noise_strengths:
            local_distances = []
            org_class_count = 0
            adv_class_count = 0
            other_class_count = 0

            for repeat in range(repeats):
                # Noise image before the attack - the attacker is aware of the noise.
                noised_image = noise_func(image, noise_strength)
                noised_image = noised_image.astype(np.float32)

                noise_predictions = model.predictions(noised_image)
                noise_prediction = np.argmax(noise_predictions)
                noised_image = np.clip(noised_image, a_min=bounds[0],
                                       a_max=bounds[1])
                # if noise_prediction != label:
                #     continue  # don't use the misclassified noisy image

                if attack.name() == 'CarliniWagnerL2Attack':
                    adversarial_image = attack(noised_image, label,
                                               max_iterations=100)
                else:
                    adversarial_image = attack(noised_image, label)

                print('adversarial image min, max:', adversarial_image.min(),
                      adversarial_image.max())

                adversarial_predictions = model.predictions(adversarial_image)
                adversarial_prediciton = np.argmax(adversarial_predictions)
                print("adversarial prediction: ", adversarial_prediciton)
                if adversarial_prediciton != label:
                    adversarial_count += 1

                noised_image = noise_func(adversarial_image, noise_strength)
                noised_image = noised_image.astype(np.float32)

                noise_predictions = model.predictions(noised_image)
                noise_prediction = np.argmax(noise_predictions)
                if noise_prediction == label:
                    org_class_count += 1
                elif noise_prediction == adversarial_prediciton:
                    adv_class_count += 1
                else:
                    other_class_count += 1

                deltas = noised_image - image
                l2_dist = np.sqrt(np.sum(deltas * deltas))
                local_distances.append(l2_dist)

            distances.append(np.mean(local_distances))
            org_class_counts.append(org_class_count)
            adv_class_counts.append(adv_class_count)
            other_class_counts.append(other_class_count)

        values = [org_class_counts, adv_class_counts, other_class_counts]
        values_names = ['original class', 'adversarial class', 'other class']
        print('distances: ', distances)
        for i in range(len(values)):
            if attack.name() == 'EmptyAttack' and i == 1:
                continue

            plt.plot(distances, values[i],
                     label=values_names[i], marker=markers[i],
                     markersize=15,
                     lw=lw, color=colors[i], linestyle=linestyles[i])

    plt.grid()
    plt.legend(loc='center right', ncol=ncols_legend, frameon=False,
               prop={'size': legend_size},
               # bbox_to_anchor=dataset[bbox]
               )

    plt.xlabel('L2 distance')
    plt.ylabel('Frequency count')
    plt.title(attack.name(), fontsize=title_size)

plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.2)
plt.savefig('channel_robustness_histograms_many_images.pdf',
            bbox_inches='tight')
plt.show()
plt.close()

print(f"\nBase test accuracy of the model: "
      f"{original_count / recover_count}")
print(f"\nAccuracy of the model after attack: "
      f"{adversarial_count / recover_count}")
print(f"\nAccuracy of the model after noising: "
      f"{defended_count / recover_count}")
