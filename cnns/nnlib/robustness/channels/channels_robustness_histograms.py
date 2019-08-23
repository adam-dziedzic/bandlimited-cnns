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
import os
import torch
import torchvision.models as models
import numpy as np
import foolbox
from foolbox.attacks import CarliniWagnerL2Attack
from foolbox.attacks import FGSM
from foolbox.attacks import L1BasicIterativeAttack
from foolbox.attacks import RandomStartProjectedGradientDescentAttack
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
from cnns.nnlib.attacks.fft_attack import FFTMultipleFrequencyAttack
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.mnist.mnist import get_mnist
from cnns.nnlib.robustness.main_adversarial import get_fmodel
from cnns.nnlib.utils.model_utils import load_model
from cnns.nnlib.datasets.cifar import cifar_mean_array, cifar_std_array
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance
import time
from cnns.nnlib.utils.general_utils import softmax
from cnns.nnlib.utils.general_utils import topk
from foolbox.criteria import TargetClass, Misclassification
from cnns.nnlib.datasets.imagenet.imagenet_from_class_idx_to_label import \
    imagenet_from_class_idx_to_label
from cnns.nnlib.datasets.cifar10_from_class_idx_to_label import \
    cifar10_from_class_idx_to_label
from cnns.nnlib.attacks.empty import EmptyAttack

beg = time.time()
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

# Instantiate the model.
# dataset = 'cifar10'
args = get_args()
args.dataset = 'imagenet'
# image_index = 235
# image_index = 249
image_index = 11754

args.use_foolbox_data = False
if args.use_foolbox_data:
    if args.dataset == 'imagenet':
        resnet = models.resnet50(pretrained=True)
        resnet.to(device)
        resnet.eval()
        net = resnet
        bounds = (0, 1)
        args.num_classes = 1000
        from_class_idx_to_label = imagenet_from_class_idx_to_label
        preprocessing = (imagenet_mean_array, imagenet_std_array)
        images, labels = foolbox.utils.samples("imagenet",
                                               data_format="channels_first",
                                               batchsize=recover_count)
    elif args.dataset == 'cifar10':
        # Use get_cifar to set the args to load the model for ciar.
        get_cifar(args, args.dataset)
        model_path = 'saved_model_2019-05-16-11-37-45-415722-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.56-channel-vals-0.model'
        net = load_model(args)
        from_class_idx_to_label = cifar10_from_class_idx_to_label
        preprocessing = (cifar_mean_array, cifar_std_array)
        images, labels = foolbox.utils.samples("cifar10",
                                               data_format="channels_first",
                                               batchsize=recover_count)

    images = images / 255  # map from [0,255] to [0,1] range
    image = images[image_index]
    label = labels[image_index]

    model = foolbox.models.PyTorchModel(
        net, bounds=bounds, num_classes=args.num_classes,
        preprocessing=preprocessing)
else:
    if args.dataset == "imagenet":
        train_loader, test_loader, train_dataset, test_dataset = load_imagenet(
            args)
        limit = 50000
    elif args.dataset == "cifar10":
        train_loader, test_loader, train_dataset, test_dataset = get_cifar(
            args, args.dataset)
        limit = 10000
    elif args.dataset == "mnist":
        train_loader, test_loader, train_dataset, test_dataset = get_mnist(
            args)
        limit = 10000
    else:
        raise Exception(f"Unknown dataset {args.dataset}")
    image, label = test_dataset.__getitem__(image_index)
    image = image.numpy()

    model, from_class_idx_to_label = get_fmodel(args=args)

print("\nimage index: ", image_index)
print("true label: ", label)

fontsize = 25
legend_size = 20
title_size = 25
font = {'size': fontsize}
import matplotlib

matplotlib.rc('font', **font)

width = 50
height = 100
fig = plt.figure(figsize=(width, height))

# target_class = 22
target_class = 282
criterion = TargetClass(target_class=target_class)
# criterion = Misclassification()

if criterion.name() == "TargetClass":
    print(f'target class id: {target_class}')
    print(f'target class name: {from_class_idx_to_label[target_class]}')
else:
    target_class = ''

fgsm_attack = FGSM(model, criterion=criterion)
bl1_attack = L1BasicIterativeAttack(model, criterion=criterion)
cw_attack = CarliniWagnerL2Attack(model, criterion=criterion)
pgd_attack = RandomStartProjectedGradientDescentAttack(model,
                                                       criterion=criterion)
empty_attack = EmptyAttack()
fft_attack = FFTMultipleFrequencyAttack(args=args, )

repeats = 100
noise_func = gauss
# noise_strengths = [0.0, 0.003, 0.1]
# noise_strengths = [0.0, 0.001, 0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08,
#                    0.09]
# noise_strengths = [0.0, 0.001, 0.007, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08,
#                    0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6,
#                    0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
#                    1.9, 2.0]
noise_strengths = [0.0]
noise_strengths += [x / 1000 for x in range(1, 10)]
noise_strengths += [x / 100 for x in range(1, 10)]
noise_strengths += [x / 10 for x in range(1, 10)]
noise_strengths += [x for x in range(1, 10)]
noise_strengths += [10.0]
# noise_strengths = []
# index = 11


# show_image(image)

# Original prediction of the model (without any adversarial changes or noise).
original_predictions = model.predictions(image)
original_prediction = np.argmax(original_predictions)
print("original prediction: ", original_prediction)
if original_prediction == label:
    original_count += 1

# attacks = [fgsm_attack, bl1_attack]
# attacks = [cw_attack]
# attacks = [pgd_attack]
# attacks = [fft_attack]
attacks = [empty_attack]

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
graph_type = 'counts'  # 'counts' or 'softs'

for attack_iter, attack in enumerate(attacks):
    if attack.name() == 'CarliniWagnerL2Attack':
        small_dist = False
        if small_dist:
            max_iterations = 1000
            binary_search_steps = 6
            initial_const = 0.01
        else:
            max_iterations = 100
            binary_search_steps = 2
            initial_const = 1e+6
            # initial_const = 1e+6

        full_name = [attack.name(), max_iterations, binary_search_steps,
                     initial_const, original_prediction, target_class]
        full_name = '-'.join([str(x) for x in full_name])

        if os.path.exists(full_name):
            adversarial_image = np.load(full_name)
        else:
            adversarial_image = attack(
                image, label,
                max_iterations=max_iterations,
                binary_search_steps=binary_search_steps,
                initial_const=initial_const,
            )
            if adversarial_image is not None:
                np.save(file=full_name, arr=adversarial_image)
    elif attack.name() == 'FFTMultipleFrequencyAttack':
        full_name = "../saved-FFT-imagenet-rounded-fft-img-idx-249-graph-recover-AttackType.RECOVERY-gauss-FFTMultipleFrequencyAttack.npy"
        # full_name = "../2019-08-21-17-20-imagenet-rounded-fft-img-idx-249-graph-recover-AttackType.RECOVERY-gauss-FFTMultipleFrequencyAttack.npy" # dist: 0.943
        # full_name = "../dist-0.48-imagenet-rounded-fft-img-idx-249-graph-recover-AttackType.RECOVERY-gauss-FFTMultipleFrequencyAttack.npy" # dist: 0.469
        adversarial_image = np.load(file=full_name)
    else:
        adversarial_image = attack(image, label)

    print('adversarial image min, max:', adversarial_image.min(),
          adversarial_image.max())

    measure = DenormDistance(mean_array=args.mean_array,
                             std_array=args.std_array)
    adv_l2_dist = measure.measure(image, adversarial_image)
    print(f'distance from adversarial to original image: {adv_l2_dist}')

    adversarial_predictions = model.predictions(adversarial_image)
    adversarial_prediction = np.argmax(adversarial_predictions)
    print("adversarial prediction: ", adversarial_prediction)
    if adversarial_prediction == label:
        adversarial_count += 1

    distances = []
    org_class_counts = []
    adv_class_counts = []
    other_class_counts = []

    org_class_softs = []
    adv_class_softs = []
    other_class_softs = []

    for noise_strength in noise_strengths:
        local_distances = []

        org_class_count = 0
        adv_class_count = 0
        other_class_count = 0

        org_class_soft = []
        adv_class_soft = []
        other_class_soft = []

        for repeat in range(repeats):
            noised_image = noise_func(adversarial_image, noise_strength)
            noised_image = noised_image.astype(np.float32)

            noise_predictions = model.predictions(noised_image)
            soft_preds = softmax(noise_predictions)
            top3 = topk(soft_preds, k=3)
            noise_prediction = top3[0]
            print('noise prediction: ', noise_prediction,
                  from_class_idx_to_label[noise_prediction])

            if graph_type == 'counts':
                if noise_prediction == original_prediction:
                    org_class_count += 1
                elif noise_prediction == adversarial_prediction:
                    adv_class_count += 1
                else:
                    other_class_count += 1
            elif graph_type == 'softs':
                org_class_soft.append(soft_preds[original_prediction])
                adv_class_soft.append(soft_preds[adversarial_prediction])
                for k in top3:
                    if k not in (adversarial_prediction, original_prediction):
                        other_class_soft.append(soft_preds[k])
                        break
            else:
                raise Exception(f"Unknown graph_type: {graph_type}")

            # deltas = noised_image - image
            # l2_dist = np.sqrt(np.sum(deltas * deltas))
            l2_dist = measure.measure(image, noised_image)
            local_distances.append(l2_dist)

        distances.append(np.mean(local_distances))
        if graph_type == 'counts':
            org_class_counts.append(org_class_count)
            adv_class_counts.append(adv_class_count)
            other_class_counts.append(other_class_count)
        elif graph_type == 'softs':
            org_class_softs.append(np.mean(org_class_soft))
            adv_class_softs.append(np.mean(adv_class_soft))
            other_class_softs.append(np.mean(other_class_soft))
        else:
            raise Exception(f"Unknown graph_type: {graph_type}")

    plt.subplot(nrows, ncols, attack_iter + 1)

    if graph_type == 'counts':
        values = [org_class_counts, adv_class_counts, other_class_counts]
    elif graph_type == 'softs':
        values = [org_class_softs, adv_class_softs, other_class_softs]
    else:
        raise Exception(f"Unknown graph_type: {graph_type}")
    values_names = ['original', 'adversarial', 'other']
    print('distances: ', distances)
    for i in range(len(values)):
        if attack.name() == 'EmptyAttack' and i == 1:
            continue
        plt.plot(distances, values[i],
                 label=values_names[i], marker=markers[i], markersize=15,
                 lw=lw, color=colors[i], linestyle=linestyles[i])

    plt.axvline(x=adv_l2_dist,
                # label='adv. dist.',
                color=get_color(MY_BLACK),
                linestyle='-')
    plt.grid()

    plt.xlabel('L2 distance')
    plt.xscale('log')
    if graph_type == 'counts':
        plt.ylabel('Frequency count')
        plt.legend(loc='center right', ncol=ncols_legend, frameon=False,
                   prop={'size': legend_size},
                   # title='class:',
                   # bbox_to_anchor=dataset[bbox]
                   )
    else:
        plt.ylim(0.0, 1.0)
        plt.ylabel('$F_y(x + \epsilon\eta)$')
        plt.legend(loc='upper center', ncol=3, frameon=False,
                   prop={'size': legend_size},
                   # title='class:',
                   # bbox_to_anchor=dataset[bbox]
                   )
    plt.title(attack.name() + ' (adv. dist.: {:.3f})'.format(adv_l2_dist),
              fontsize=title_size)

plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.2)
plt.savefig(
    'graphs/' + get_log_time() + '-' + attack.name() + '-channel_robustness_histograms.pdf',
    bbox_inches='tight')
plt.show()
plt.close()

print(f"\nBase test accuracy of the model: "
      f"{original_count / recover_count}")
print(f"\nAccuracy of the model after attack: "
      f"{adversarial_count / recover_count}")
print(f"\nAccuracy of the model after noising: "
      f"{defended_count / recover_count}")
print(f"total elapsed time: {time.time() - beg}")
