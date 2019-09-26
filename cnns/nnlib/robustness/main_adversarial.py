"""
Use the rounding and fft pre-processing and find adversarial examples after
such transformations.
"""

# If you use Jupyter notebooks uncomment the line below
# %matplotlib inline

# Use the import below to run the code remotely on a server.
from cnns import matplotlib_backend

print('Using: ', matplotlib_backend.backend)

import matplotlib

print('Using: ', matplotlib.get_backend())

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

import time
import pickle
import os
import foolbox
import numpy as np
import torch
from random import sample
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.attacks.carlini_wagner_round_fft import \
    CarliniWagnerL2AttackRoundFFT
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.datasets.transformations.normalize import Normalize
from cnns.nnlib.datasets.transformations.denormalize import Denormalize
from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance
from cnns.nnlib.utils.complex_mask import get_hyper_mask
from foolbox.attacks.additive_noise import AdditiveUniformNoiseAttack
from foolbox.attacks.additive_noise import AdditiveGaussianNoiseAttack
from cnns.nnlib.utils.object import Object
from cnns.nnlib.robustness.utils import to_fft
from cnns.nnlib.robustness.utils import laplace_noise
from cnns.nnlib.robustness.randomized_defense import defend
import matplotlib
from cnns.nnlib.pytorch_layers.fft_band_2D import FFTBandFunction2D
from cnns.nnlib.datasets.transformations.denorm_round_norm import \
    DenormRoundNorm
from cnns.nnlib.utils.svd2d import compress_svd
from cnns.nnlib.utils.general_utils import AdversarialType
from cnns.nnlib.attacks.guass_attack import GaussAttack
from cnns.nnlib.attacks.fft_attack import FFTHighFrequencyAttackAdversary
from cnns.nnlib.attacks.fft_attack import FFTHighFrequencyAttack
from cnns.nnlib.attacks.fft_attack import FFTLimitFrequencyAttack
from cnns.nnlib.attacks.fft_attack import FFTLimitFrequencyAttackAdversary
from cnns.nnlib.attacks.fft_attack import FFTReplaceFrequencyAttack
from cnns.nnlib.attacks.fft_attack import FFTSingleFrequencyAttack
from cnns.nnlib.attacks.fft_attack import FFTMultipleFrequencyAttack
from cnns.nnlib.attacks.fft_attack import FFTSmallestFrequencyAttack
from cnns.nnlib.attacks.fft_attack import FFTLimitValuesAttack
from cnns.nnlib.attacks.fft_attack import FFTLimitMagnitudesAttack
from cnns.nnlib.attacks.fft_attack import FFTMultipleFrequencyBinarySearchAttack
from cnns.nnlib.attacks.nattack import Nattack
from cnns.nnlib.attacks.empty import EmptyAttack
from cnns.nnlib.utils.general_utils import softmax
from cnns.nnlib.robustness.channels.channels_definition import \
    gauss_noise_fft_torch
from foolbox.criteria import TargetClass, Misclassification
from cnns.nnlib.attacks.simple_blackbox_attack import SimbaSingle
from cnns.nnlib.robustness.gradients.compute import compute_gradients
from cnns.nnlib.robustness.fmodel import get_fmodel
from cnns.nnlib.datasets.load_data import get_data
results_folder = "results/"
delimiter = ";"

font = {'family': 'normal',
        'size': 18}

matplotlib.rc('font', **font)

adv_images = []
adv_labels = []
adv_recovered_images = []
adv_recovered_labels = []

org_images = []
org_labels = []
org_recovered_images = []
org_recovered_labels = []

gauss_images = []
gauss_labels = []
gauss_recovered_images = []
gauss_recovered_labels = []
gauss_non_recovered_images = []
guass_non_recovered_images = []


def save_file(name, images, labels):
    save_time = get_log_time() + '-len-' + str(len(labels))
    pickle_protocol = 2
    file_name = save_time + '-' + name + '-images'
    with open(file_name, 'wb') as f:
        data = {'images': images, 'labels': labels}
        pickle.dump(data, f, pickle_protocol)


def save_adv_org():
    if args.save_out:
        save_data = []
        save_data.append(['adv', adv_images, adv_labels])
        save_data.append(['org', org_images, org_labels])
        save_data.append(['gauss', gauss_images, gauss_labels])
        save_data.append(
            ['adv_recovered', adv_recovered_images, adv_recovered_labels])
        save_data.append(
            ['org_recovered', org_recovered_images, org_recovered_labels])
        save_data.append(
            ['gauss_recovered', gauss_recovered_images, gauss_recovered_labels])
        for name, images, labels in save_data:
            save_file(name=name + '-' + args.dataset, images=images,
                      labels=labels)


def print_heat_map(input_map, args, title="", ylabel=""):
    args.plot_index += 1
    plt.subplot(args.rows, args.cols, args.plot_index)
    if args.plot_index % args.cols == 1:
        plt.ylabel(ylabel, fontsize=20)
    plt.title(title)
    # input_map = np.clip(input_map, a_min=0, a_max=80)
    is_legend = True
    if is_legend:
        cmap = "hot"  # matplotlib.cm.gray  # "hot"
        plt.imshow(input_map, cmap=cmap, interpolation='nearest')
        plt.colorbar()
        # heatmap_legend = plt.pcolor(input_map)
        # plt.colorbar(heatmap_legend)
    else:
        # cmap = matplotlib.cm.gray
        cmap = "hot"
        # cmap = "Spectral"
        # cmap = "seismic"
        plt.imshow(input_map, cmap=cmap, interpolation='nearest')
    # plt.axis('off')


def get_cmap(args):
    lim_y, lim_x = args.init_y, args.init_x
    args.lim_y = lim_y
    args.lim_x = lim_x
    # lim_y, lim_x = init_y // 2, init_x // 2
    # lim_y, lim_x = 2, 2
    # lim_y, lim_x = 3, 3

    cmap_type = "matshow"  # "standard" or "custom"
    # cmap_type = "standard"

    # vmin_heatmap = -6
    # vmax_heatmap = 10

    vmin_heatmap = None
    vmax_heatmap = None

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
    return cmap, cmap_type, vmin_heatmap, vmax_heatmap


def print_color_map(x, args, fig=None, ax=None):
    cmap, cmap_type, vmin_heatmap, vmax_heatmap = get_cmap(args)
    decimals = args.decimals

    map_labels = "None"  # "None" or "Text"
    # map_labels = "Text"

    if cmap_type == "standard":
        plt.imshow(x, cmap=cmap,
                   interpolation=args.interpolation)
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


def get_L2_dist(input1, input2):
    diff = input1 - input2
    return np.sqrt(np.sum(diff * diff))


def replace_frequency(original_image, images, labels, attack_fn, args,
                      high=True):
    true_label = args.True_class_id
    net = args.fmodel.predictions
    last_adv_image = None
    original_image2 = None
    # We want to find the target image from which we copy frequency to the
    # original image to make it adversarial such that the adversarial image has
    # the smallest L2 distance to the original image.
    adv_min_L2_dist = float('inf')
    # size = len(images)
    # indexer = range(10000, 40000, 1)
    # indexer = range(150, 250, 1)
    indexer = sample(range(0, 10000), 10)
    for idx in indexer:
        begin = time.time()
        if labels is None:
            image, target_label_id = images.__getitem__(idx)
            image = image.numpy()
        else:
            # For the dataset from the foolbox library.
            image = images[idx]
            target_label_id = labels[idx]
        # If targeted attack, select only images with target label.
        if args.targeted_attack and target_label_id != args.target_lagel_id:
            continue
        # If un-targeted attack, select only images with label different than the
        # ground truth label (for the original image).
        if (not args.targeted_attack) and target_label_id == true_label:
            continue
        # Accept only the target image that is classified correctly.
        label_id = np.argmax(net(image))
        if target_label_id == label_id:
            # check:
            # cnns.nnlib.robustness.attacks.fft_attack.FFTReplaceFrequencyAttack
            adv_image = attack_fn(
                original_image, input2=image, label=true_label,
                target_label=target_label_id, net=net, high=high)
            if adv_image is not None:
                l2_dist = get_L2_dist(input1=original_image, input2=image)
                if l2_dist < adv_min_L2_dist:
                    adv_min_L2_dist = l2_dist
                    original_image2 = image
                    last_adv_image = adv_image
        print('index: ', idx, ' elapsed time: ', time.time() - begin)
    # original_image2 = original_image2.astype(np.float32)
    # Normalize the data for the pytorch models.
    # original_image2 = args.normalizer.normalize(original_image2)

    return last_adv_image, original_image2


def classify_image(image, original_image, args, title="",
                   clip_input_image=False, ylabel_text="Original image",
                   is_denormalize=True):
    result = Object()
    predictions = args.fmodel.predictions(image=image)
    soft_predictions = softmax(predictions)
    predicted_class_id = np.argmax(soft_predictions)
    predicted_label = args.from_class_idx_to_label[predicted_class_id]
    confidence = np.max(soft_predictions)

    result.confidence = confidence
    result.label = predicted_label
    result.class_id = predicted_class_id
    result.L2_distance = args.meter.measure(original_image, image)
    result.L1_distance = args.meter.measure(original_image, image, norm=1)
    result.Linf_distance = args.meter.measure(original_image, image,
                                              norm=float('inf'))

    if args.is_debug:
        # Set the lim_y and lim_x in args.
        get_cmap(args=args)
        true_label = args.true_label
        print("Number of unique values: ", len(np.unique(image)))
        print(title + ": model predicted label: ", predicted_label)
        if predicted_label != true_label:
            print(f"The True label: {true_label} is different than "
                  f"the predicted label: {predicted_label}")
        print(title)
        title_str = title
        # title_str += ' label: ' + str(
        #     predicted_label.replace(",", "\n")) + "\n"
        title_str += ' label: ' + str(predicted_label) + "\n"
        confidence_str = str(np.around(confidence, decimals=args.decimals))
        title_str += "confidence: " + confidence_str + "\n"
        if id(image) != id(original_image):
            pass
            # title_str += "L1 distance: " + str(
            #     result.L1_distance) + "\n"
            # title_str += "L2 distance: " + str(
            #     result.L2_distance) + "\n"
            # title_str += "Linf distance: " + str(
            #     result.Linf_distance) + "\n"
        L2 = str(np.around(result.L2_distance, decimals=args.decimals))
        title_str += "L2 distance: " + str(L2) + "\n"
        # ylabel_text = "spatial domain"
        image_show = image
        if clip_input_image:
            # image = torch.clamp(image, min = args.min, max=args.max)
            image_show = np.clip(image_show, a_min=args.min,
                                 a_max=args.max)
        if is_denormalize:
            # Go to the 0-1 range.
            image_show = args.denormalizer.denormalize(image_show)
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
            plt.subplot(args.rows, args.cols, args.plot_index)
            # title_str = "Spatial domain"
            plt.title(title_str)
            plt.imshow(
                np.moveaxis(image_show, 0, -1),
                # move channels to last dimension
                cmap=args.cmap)
            # plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
            # plt.axis('off')
            if args.plot_index % args.cols == 1:
                plt.ylabel(ylabel_text)

    return result


def print_fft(image, channel, args, title="", is_log=True):
    print("fft: ", title)
    print("input min: ", image.min())
    print("input max: ", image.max())
    image = np.clip(image, a_min=-2.12, a_max=2.64)
    xfft = to_fft(image, fft_type=args.fft_type, is_log=is_log,
                  is_DC_shift=True)
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
    xfft = xfft[:args.lim_y, :args.lim_x]
    # print("original_fft:\n", original_fft)
    # print_color_map(xfft, fig, ax)
    ylabel = "fft-ed channel " + str(channel) + ":\n" + args.fft_type
    print_heat_map(xfft, args=args, title=title, ylabel=ylabel)

    return xfft


def roundfft_recover(result, image, original_image, attack_round_fft):
    if args.values_per_channel > 0 and args.compress_fft_layer > 0 and image is not None:
        rounder = DenormRoundNorm(
            mean_array=args.mean_array, std_array=args.std_array,
            values_per_channel=args.values_per_channel)
        round_image = rounder.round(np.copy(image))
        roundfft_image = attack_round_fft.fft_complex_compression(
            image=np.copy(round_image))
        result_roundfft = classify_image(
            image=roundfft_image,
            original_image=original_image,
            args=args)
        result.add(result_roundfft, prefix="roundfft_")
    else:
        result.roundfft_label = None


def fftround_recover(result, image, original_image, attack_round_fft):
    if args.values_per_channel > 0 and args.compress_fft_layer > 0 and image is not None:
        fft_image = attack_round_fft.fft_complex_compression(
            image=np.copy(image))
        rounder = DenormRoundNorm(
            mean_array=args.mean_array, std_array=args.std_array,
            values_per_channel=args.values_per_channel)
        fftround_image = rounder.round(np.copy(fft_image))
        result_fftround = classify_image(
            image=fftround_image,
            original_image=original_image,
            args=args)
        result.add(result_fftround, prefix="fftround_")
    else:
        result.fftround_label = None


def fftuniform_recover(result, image, original_image, attack_round_fft):
    if args.noise_epsilon > 0 and args.compress_fft_layer > 0 and image is not None:
        fft_image = attack_round_fft.fft_complex_compression(
            image=np.copy(image))
        noise = AdditiveUniformNoiseAttack()._sample_noise(
            epsilon=args.noise_epsilon, image=fft_image,
            bounds=(args.min, args.max))
        fftuniform_image = fft_image + noise
        result_fftuniform = classify_image(
            image=fftuniform_image,
            original_image=original_image,
            args=args)
        result.add(result_fftuniform, prefix="fftuniform_")
    else:
        result.fftuniform_label = None


def roundsvd_recover(result, image, original_image):
    if args.values_per_channel > 0 and args.svd_compress > 0 and image is not None:
        rounder = DenormRoundNorm(
            mean_array=args.mean_array, std_array=args.std_array,
            values_per_channel=args.values_per_channel)
        round_image = rounder.round(np.copy(image))
        roundsvd_image = compress_svd(
            torch_img=torch.tensor(np.copy(round_image)),
            compress_rate=args.svd_compress)
        roundsvd_image = roundsvd_image.cpu().numpy()
        result_roundsvd = classify_image(
            image=roundsvd_image,
            original_image=original_image,
            args=args)
        result.add(result_roundsvd, prefix="roundsvd_")
    else:
        result.roundsvd_label = None


def rounduniform_recover(result, image, original_image):
    if args.values_per_channel > 0 and args.noise_epsilon > 0 and image is not None:
        rounder = DenormRoundNorm(
            mean_array=args.mean_array, std_array=args.std_array,
            values_per_channel=args.values_per_channel)
        round_image = rounder.round(np.copy(image))
        noise = AdditiveUniformNoiseAttack()._sample_noise(
            epsilon=args.noise_epsilon, image=round_image,
            bounds=(args.min, args.max))
        rounduniform_image = round_image + noise
        result_rounduniform = classify_image(
            image=rounduniform_image,
            original_image=original_image,
            args=args)
        result.add(result_rounduniform, prefix="rounduniform_")
    else:
        result.rounduniform_label = None


def run(args):
    result = Object()
    fmodel, pytorch_model, from_class_idx_to_label = get_fmodel(args=args)

    args.fmodel = fmodel
    args.from_class_idx_to_label = from_class_idx_to_label
    args.decimals = 4  # How many digits to print after the decimal point.

    # Choose how many channels should be plotted.
    channels_nr = 1

    criterion_name = 'Misclassifiction'  # 'TargetClass' or 'Misclassification'
    # criterion_name = 'TargetClass'

    if args.targeted_attack:
        target_class = 282
        criterion = TargetClass(target_class=target_class)
        print(f'target class id: {target_class}')
        print(f'target class name: {from_class_idx_to_label[target_class]}')
    else:
        criterion = Misclassification()
        target_class = ''
        print('No target class specified')

    # Choose what fft types should be plotted.
    # fft_types = ["magnitude"]
    # fft_types = ["magnitude", "phase"]
    fft_types = []
    if args.is_debug:
        # fft_types = ["magnitude"]
        fft_types = []

    # channels = [x for x in range(channels_nr)]
    channels = [0]
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
        attack = foolbox.attacks.CarliniWagnerL2Attack(
            fmodel, criterion=criterion)
    elif args.attack_name == "CarliniWagnerL2AttackRoundFFT":
        # L2 norm
        attack = CarliniWagnerL2AttackRoundFFT(model=fmodel, args=args,
                                               get_mask=get_hyper_mask)
    elif args.attack_name == "ProjectedGradientDescentAttack":
        # L infinity norm
        attack = foolbox.attacks.ProjectedGradientDescentAttack(
            fmodel, criterion=criterion)
    elif args.attack_name == "FGSM":
        # L infinity norm
        attack = foolbox.attacks.FGSM(fmodel, criterion=criterion)
    elif args.attack_name == "RandomStartProjectedGradientDescentAttack":
        attack = foolbox.attacks.RandomStartProjectedGradientDescentAttack(
            fmodel, criterion=criterion)
    elif args.attack_name == "DeepFoolAttack":
        # L2 attack by default, can also be L infinity
        attack = foolbox.attacks.DeepFoolAttack(fmodel, criterion=criterion)
    elif args.attack_name == "LBFGSAttack":
        attack = foolbox.attacks.LBFGSAttack(fmodel, criterion=criterion)
    elif args.attack_name == "L1BasicIterativeAttack":
        attack = foolbox.attacks.L1BasicIterativeAttack(
            fmodel, criterion=criterion)
    elif args.attack_name == "GaussAttack":
        attack = GaussAttack()
    elif args.attack_name == "FFTHighFrequencyAttack":
        attack = FFTHighFrequencyAttack()
    elif args.attack_name == "FFTHighFrequencyAttackAdversary":
        attack = FFTHighFrequencyAttackAdversary()
    elif args.attack_name == "FFTLimitFrequencyAttack":
        attack = FFTLimitFrequencyAttack()
    elif args.attack_name == "FFTLimitFrequencyAttackAdversary":
        attack = FFTLimitFrequencyAttackAdversary()
    elif args.attack_name == "FFTReplaceFrequencyAttack":
        attack = FFTReplaceFrequencyAttack()
    elif args.attack_name == "FFTSingleFrequencyAttack":
        attack = FFTSingleFrequencyAttack(fmodel)
    elif args.attack_name == "FFTMultipleFrequencyAttack":
        attack = FFTMultipleFrequencyAttack(
            args=args,
            model=fmodel,
            max_frequencies_percent=10,
            iterations=100,
        )
    elif args.attack_name == "FFTMultipleFrequencyBinarySearchAttack":
        attack = FFTMultipleFrequencyBinarySearchAttack(model=fmodel, args=args)
    elif args.attack_name == "FFTSmallestFrequencyAttack":
        attack = FFTSmallestFrequencyAttack(fmodel)
    elif args.attack_name == "FFTLimitValuesAttack":
        attack = FFTLimitValuesAttack(fmodel)
    elif args.attack_name == "FFTLimitMagnitudesAttack":
        attack = FFTLimitMagnitudesAttack(fmodel)
    elif args.attack_name == "Nattack":
        # We operate directly in PyTorch.
        attack = Nattack(model=pytorch_model, args=args)
    elif args.attack_name == 'SimbaSingle':
        attack = SimbaSingle(model=pytorch_model, args=args)
    elif args.attack_name == "EmptyAttack":
        attack = EmptyAttack(fmodel)
    elif args.attack_name is None or args.attack_name == 'None':
        print("No attack set!")
        attack = None
    else:
        raise Exception("Unknown attack name: " + str(args.attack_name))
    attacks = [attack]

    show_diff_spatial = False
    if args.is_debug:
        # 1 is for the first row of images.
        rows = len(attacks) * (1 + len(fft_types) * len(channels))
        cols = 1  # print at least the original image
        if args.values_per_channel > 0:
            cols += 1
        if args.compress_fft_layer > 0:
            cols += 1
        if args.noise_sigma > 0:
            cols += 1
        if args.noise_epsilon > 0:
            cols += 1
        if args.laplace_epsilon > 0:
            cols += 1
        if args.svd_compress > 0:
            cols += 1
        if args.adv_type != AdversarialType.NONE:
            cols += 1
        show_diff = False
        if show_diff:
            col_diff = 2
            cols += col_diff
        show_diff_spatial = True
        if show_diff_spatial:
            cols += 1
        show_2nd = False  # show 2nd image
        if show_2nd:
            # Should we show the diffs?
            show_2nd_col_diff = False
            if show_2nd_col_diff:
                col_diff2nd = 3
                cols += col_diff2nd
            else:
                cols += 1

        args.rows = rows
        args.cols = cols

        # args.rows = 1
        # args.cols = 2
        plt.figure(figsize=(cols * 10, rows * 12))
        # This returns the fig object.
        # plt.figure()

        # index for each subplot
        args.plot_index = 0
    else:
        show_2nd = False
        show_diff = False

    if False:
        args.rows = 3
        args.cols = 2
        plt.figure(figsize=(cols * 2, rows * 21))

    args.normalizer = Normalize(mean_array=args.mean_array,
                                std_array=args.std_array)
    args.denormalizer = Denormalize(mean_array=args.mean_array,
                                    std_array=args.std_array)
    args.meter = DenormDistance(mean_array=args.mean_array,
                                std_array=args.std_array)

    if args.use_foolbox_data:
        images, labels = foolbox.utils.samples(dataset=args.dataset, index=0,
                                               batchsize=20,
                                               # batchsize=3,
                                               shape=(args.init_y, args.init_x),
                                               data_format='channels_first')
        # print("max value in images pixels: ", np.max(images))
        images = images / 255
        # print("max value in images after 255 division: ", np.max(images))
    else:
        images = test_dataset
        # images = train_dataset
        labels = None

    # for attack_strength in args.attack_strengths:
    for attack in attacks:
        # get source image and label, args.idx - is the index of the image
        # image, label = foolbox.utils.imagenet_example()
        if args.use_foolbox_data:
            original_image, args.True_class_id = images[args.image_index], \
                                                 labels[
                                                     args.image_index]
            original_image = original_image.astype(np.float32)

            # normalize the data for the pytorch models.
            original_image = args.normalizer.normalize(original_image)
        else:
            original_image, args.True_class_id = test_dataset.__getitem__(
                args.image_index)
            original_image = original_image.numpy()

        print('min : max value in image pixels: ', np.min(original_image), ',',
              np.max(original_image))
        # np.set_printoptions(threshold=2**30)
        # print(original_image)
        print(args.True_class_id)
        # np.save(file='imagenet_249' + ".npy", arr=original_image)

        original_image2 = None

        if args.dataset == "mnist":
            args.True_class_id = args.True_class_id.item()

        args.true_label = from_class_idx_to_label[args.True_class_id]
        result.true_label = args.true_label
        result.true_id = args.True_class_id
        print("True class id:", args.True_class_id, ", is label: ",
              args.true_label)

        # original image.
        if args.show_original:
            original_title = "original"
            original_result = classify_image(
                image=original_image,
                original_image=original_image,
                args=args,
                title=original_title)
            result.add(original_result, prefix="original_")
            print("label for the original image: ", original_result.label)

        image = original_image

        # from_file = False
        full_name = args.dataset
        if args.use_foolbox_data:
            # full_name += "-compress-fft-layer-0"
            full_name += "-foolbox"
        else:
            full_name += "-rounded-fft"
        full_name += "-img-idx-" + str(args.image_index) + "-graph-recover"

        if args.is_debug:
            pass
            # full_name = str(
            #     args.noise_iterations) + "-" + full_name + "-" + get_log_time()
            # full_name += "-fft-network-layer"
            # full_name += "-fft-only-recovery"
            full_name += "-" + str(args.attack_type) + "-" + str(
                args.recover_type)  # get_log_time()

        adv_image = None
        created_new_adversarial = False
        # The image has to be correctly classified in the first place to then
        # create an adversarial example for it.
        if args.adv_type == AdversarialType.BEFORE and (
                original_result.class_id == args.True_class_id):
            attack_name = attack.name()
            print("attack name: ", attack_name)
            # if attack_name != "carliniwagnerl2attack":
            full_name += "-" + str(attack_name)
            if attack_name == "CarliniWagnerL2AttackRoundFFT":
                full_name += "-" + str(args.recover_type)
                full_name += "-c-val-" + str(args.attack_strength)
            print("full name of stored adversarial example: ", full_name)
            is_load_image = False
            if is_load_image and os.path.exists(full_name + ".npy") and (
                    attack_name != "CarliniWagnerL2AttackRoundFFT") and (
                    attack_name != "GaussAttack") and (
                    attack_name != "CarliniWagnerL2Attack") and (
                    attack_name != "Nattack"
            ):
                # and not (attack_name.startswith('FFT')):
                print('found image to load')
                adv_image = np.load(file=full_name + ".npy")
                result.adv_timing = -1
            else:
                start_adv = time.time()
                if attack_name.startswith("Carlini"):
                    adv_image = attack(
                        original_image, args.True_class_id,
                        # max_iterations=args.attack_max_iterations,
                        # max_iterations=2, binary_search_steps=1, initial_const=1e+12,
                        max_iterations=1000, binary_search_steps=1,
                        initial_const=args.attack_strength,
                        confidence=args.attack_confidence,
                    )
                elif attack_name == "GaussAttack":
                    adv_image = GaussAttack(original_image,
                                            bounds=(args.min, args.max),
                                            epsilon=attack_strength)
                elif attack_name == "FFTHighFrequencyAttack":
                    adv_image = attack(
                        input_or_adv=original_image,
                        label=args.True_class_id,
                        compress_rate=args.compress_rate)
                elif attmack_nae == "FFTHighFrequencyAttackAdversary":
                    adv_image = attack(
                        original_image, label=args.True_class_id,
                        net=fmodel.predictions)
                elif attack_name == "FFTLimitFrequencyAttack":
                    adv_image = attack(
                        input_or_adv=original_image,
                        label=args.True_class_id,
                        net=fmodel.predictions,
                        compress_rate=args.compress_rate)
                elif attack_name == "FFTLimitFrequencyAttackAdversary":
                    adv_image = attack(
                        input_or_adv=original_image,
                        label=args.True_class_id,
                        net=fmodel.predictions)
                elif attack_name == "FFTReplaceFrequencyAttack":
                    adv_image, original_image2 = replace_frequency(
                        original_image=original_image, images=images,
                        labels=labels, attack_fn=attack, args=args, high=False)
                elif attack_name == 'FFTMultipleFrequencyAttack':
                    # full_name = "saved-FFT-imagenet-rounded=-fft-img-idx-249-graph-recover-AttackType.RECOVERY-gauss-FFTMultipleFrequencyAttack.npy"
                    # adv_image = np.load(file=full_name)
                    adv_image = attack(input_or_adv=original_image,
                                       label=args.True_class_id)
                else:
                    adv_image = attack(input_or_adv=original_image,
                                       label=args.True_class_id)
                result.adv_timing = time.time() - start_adv
                created_new_adversarial = True

            if show_2nd and original_image2 is not None:
                result_original2 = classify_image(
                    image=original_image2,
                    original_image=original_image2,
                    args=args,
                    # title="original 2nd",
                    title="target",
                )
                result.add(result_original2, prefix="original2_")

            if args.is_debug and show_diff_spatial:
                original_01 = args.denormalizer.denormalize(original_image)
                adv_01 = args.denormalizer.denormalize(adv_image)
                diff = 10 * (original_01 - adv_01)
                classify_image(
                    image=diff,
                    original_image=original_image,
                    args=args,
                    title="difference",
                    is_denormalize=False,
                )

            if adv_image is not None:
                l2_dist_adv_original = args.meter.measure(original_image,
                                                          adv_image)
                print(
                    "l2 distance between adversarial image and the original image: ",
                    l2_dist_adv_original)
                image = adv_image
                result_adv = classify_image(
                    image=adv_image,
                    original_image=original_image,
                    args=args,
                    title="adv.")
                print("adversarial label, id: ", result_adv.label,
                      result_adv.class_id)
                print('adversarial L2 dist: ', result_adv.L2_distance)
                print('adversarial Linf dist: ', result_adv.Linf_distance)
                result.add(result_adv, prefix="adv_")
            else:
                print('the adversarial example has not been found.')
                result.adv_label = None

        # The rounded image.
        if args.values_per_channel > 0 and image is not None:
            rounder = DenormRoundNorm(
                mean_array=args.mean_array, std_array=args.std_array,
                values_per_channel=args.values_per_channel)
            rounded_image = rounder.round(np.copy(image))
            print("rounded_image min and max: ", rounded_image.min(), ",",
                  rounded_image.max())
            title = "cd (" + str(args.values_per_channel) + ")"
            result_round = classify_image(
                image=rounded_image,
                original_image=original_image,
                args=args,
                title=title)
            print("show diff between input image and rounded: ",
                  np.sum(np.abs(rounded_image - original_image)))
            result.add(result_round, prefix="round_")
            print("label, id found after applying rounding: ",
                  result_round.label, result_round.class_id)
        else:
            result.round_label = None

        if args.compress_fft_layer > 0 and image is not None:
            is_approximate = False
            if is_approximate:
                compress_image = FFTBandFunction2D.forward(
                    ctx=None,
                    input=torch.from_numpy(np.copy(image)).unsqueeze(0),
                    compress_rate=args.compress_fft_layer).numpy().squeeze(0)
            else:
                # exact compression.
                compress_image = attack_round_fft.fft_complex_compression(
                    image=np.copy(image))
            # title = "fc (" + str(args.compress_fft_layer) + ")"
            title = "spatial domain"
            # title += "interpolation: " + args.interpolate
            result_fft = classify_image(
                image=compress_image,
                original_image=original_image,
                args=args,
                ylabel_text="compressed (" + str(
                    args.compress_fft_layer) + "%)",
                title=title)
            result.add(result_fft, prefix="fft_")
            print("label, id found after applying fft: ",
                  result_fft.label, result_fft.class_id)
        else:
            result.fft_label = None

        if args.noise_sigma > 0 and image is not None:
            # gauss_image = gauss(image_numpy=image, sigma=args.noise_sigma)
            is_fft = False
            if is_fft:
                input = torch.tensor(image).unsqueeze(0)
                gauss_image = gauss_noise_fft_torch(std=args.noise_sigma,
                                                    input=input)
                gauss_image = gauss_image.detach().cpu().squeeze().numpy()
            else:
                noise = AdditiveGaussianNoiseAttack()._sample_noise(
                    epsilon=args.noise_sigma, image=image,
                    bounds=(args.min, args.max))
                gauss_image = image + noise

            l2_dist_gauss_original = args.meter.measure(gauss_image,
                                                        original_image)
            print(
                "l2 distance between gauss image and the original image: ",
                l2_dist_gauss_original)

            if adv_image is not None:
                l2_dist_gauss_adv = args.meter.measure(gauss_image, adv_image)
                print("l2 distance between gauss image and the adv image: ",
                      l2_dist_gauss_adv)

            # title = "level of gaussian-noise: " + str(args.noise_sigma)
            title = "gauss (" + str(args.noise_sigma) + ")"
            result_gauss = classify_image(
                image=gauss_image,
                original_image=original_image,
                args=args,
                title=title)
            result.add(result_gauss, prefix="gauss_")

            start_time = time.time()
            result_noise = randomized_defense(image=image, fmodel=fmodel,
                                              original_image=original_image,
                                              defense_name="gauss")
            result.time_gauss_defend = time.time() - start_time

            result.add(result_noise, prefix="gauss_many_")
            print("label, id found after applying gauss: ",
                  result_gauss.label, result_gauss.class_id)
        else:
            result.gauss_label = None

        if adv_image is not None and original_result.class_id == args.True_class_id and gauss_image:
            gradients, grad_stats = compute_gradients(args=args,
                                                      model=pytorch_model,
                                                      original_image=original_image,
                                                      original_label=original_result.class_id,
                                                      adv_image=adv_image,
                                                      adv_label=result_adv.class_id,
                                                      gauss_image=gauss_image)
            if result_adv:
                grad_stats['adv_confidence'] = result_adv.confidence
            if original_result:
                grad_stats['original_confidence'] = original_result.confidence
            if result_gauss:
                is_gauss_recovered = (
                        result_gauss.class_id == args.True_class_id)
                grad_stats['is_gauss_recovered'] = is_gauss_recovered
            else:
                raise Exception('We need the result of adding Gaussian noise.')

            grad_stats['image_index'] = args.image_index
            grad_stats['attack_strength'] = args.attack_strength

            # Write with respect to the recovered or not.
            file_recovered_name = args.global_log_time + '_grad_stats_' + str(
                    is_gauss_recovered) + '-' + args.dataset + '.csv'
            with open(file_recovered_name, 'a') as file:
                for key in sorted(grad_stats.keys()):
                    val = grad_stats[key]
                    file.write(key + ';' + str(val) + ';')
                file.write('\n')
                file.flush()

            # Write everything.
            with open(args.global_log_time + '_' + args.dataset + '_grad_stats.csv', 'a') as file:
                for key in sorted(grad_stats.keys()):
                    val = grad_stats[key]
                    file.write(key + ';' + str(val) + ';')
                file.write('\n')
                file.flush()

        if args.noise_epsilon > 0 and image is not None:
            print("uniform additive noise defense")
            l2_dist_adv_original = args.meter.measure(original_image, image)
            print(
                "l2 distance between image (potential adversarial) and "
                "original images: ",
                l2_dist_adv_original)
            # title = "level of uniform noise: " + str(args.noise_epsilon)
            title = "uniform (" + str(args.noise_epsilon) + ")"
            noise = AdditiveUniformNoiseAttack()._sample_noise(
                epsilon=args.noise_epsilon, image=image,
                bounds=(args.min, args.max))
            noise_image = image + noise
            result_noise = classify_image(
                image=noise_image,
                original_image=original_image,
                args=args,
                title=title)
            print("label, id found after applying uniform random noise once: ",
                  result_noise.label, result_noise.class_id)
            result.add(result_noise, prefix="noise_")

            start_time_defend = time.time()
            result_noise = randomized_defense(image=image, fmodel=fmodel,
                                              original_image=original_image,
                                              defense_name="uniform")
            result.time_noise_defend = time.time() - start_time_defend

            result.add(result_noise, prefix="noise_many_")
        else:
            result.noise_label = None

        if args.laplace_epsilon > 0 and image is not None:
            print("laplace noise defense")
            title = "laplace (" + str(args.laplace_epsilon) + ")"
            noise = laplace_noise(shape=image.shape,
                                  epsilon=args.laplace_epsilon,
                                  dtype=image.dtype,
                                  args=args)
            noise_image = image + noise
            result_laplace = classify_image(
                image=noise_image,
                original_image=original_image,
                args=args,
                title=title)
            print("label, id found after applying laplace noise once: ",
                  result_laplace.label, result_laplace.class_id)
            result.add(result_laplace, prefix="laplace_")

            start_time = time.time()
            result_laplace = randomized_defense(image=image, fmodel=fmodel,
                                                original_image=original_image)
            result.time_laplace_defend = time.time() - start_time

            result.add(result_laplace, prefix="laplace_many_")
        else:
            result.laplace_label = None

        if args.svd_compress > 0 and image is not None:
            print("svd defense")
            title = "svd (" + str(args.svd_compress) + ")"
            svd_image = compress_svd(torch_img=torch.tensor(np.copy(image)),
                                     compress_rate=args.svd_compress)
            svd_image = svd_image.cpu().numpy()
            print("svd image min and max: ", svd_image.min(), ",",
                  svd_image.max())
            result_svd = classify_image(
                image=svd_image,
                original_image=original_image,
                args=args,
                title=title)
            print("show diff between input image and svd image: ",
                  np.sum(np.abs(svd_image - original_image)))
            result.add(result_svd, prefix="svd_")
            print("label, id found after applying svd: ",
                  result_svd.label, result_svd.class_id)
        else:
            result.svd_label = None

        if args.recover_type == "roundfft":
            roundfft_recover(result=result, image=image,
                             original_image=original_image,
                             attack_round_fft=attack_round_fft)

        if args.recover_type == "fftround":
            fftround_recover(result=result, image=image,
                             original_image=original_image,
                             attack_round_fft=attack_round_fft)

        if args.recover_type == "fftuniform":
            fftuniform_recover(result=result, image=image,
                               original_image=original_image,
                               attack_round_fft=attack_round_fft)

        if args.recover_type == "roundsvd":
            roundsvd_recover(result=result, image=image,
                             original_image=original_image)

        if args.recover_type == "rounduniform":
            rounduniform_recover(result=result, image=image,
                                 original_image=original_image)

        if args.adv_type == AdversarialType.AFTER:
            full_name += "-after"
            print("adv_type: ", args.adv_type, " attack name: ",
                  attack.name())
            if os.path.exists(full_name + ".npy"):
                adv_image = np.load(file=full_name + ".npy")
                result.adv_timing = -1
            else:
                start_adv = time.time()
                adv_image = attack(original_image, args.True_class_id)
                result.adv_timing = time.time() - start_adv
                created_new_adversarial = True
            if adv_image is not None:
                result_adv = classify_image(
                    image=adv_image,
                    original_image=original_image,
                    args=args,
                    title="adversarial")
                result.add(result_adv, prefix="adv_")
            else:
                adv_image = None
                result.adv_label = None

        if args.adv_type == AdversarialType.NONE:
            result.adv_label = None
            adv_image = None

        if adv_image is not None and (created_new_adversarial):
            if (attack_name != "CarliniWagnerL2AttackRoundFFT") and (
                    attack_name != "GaussAttack") and not attack_name.startswith(
                'FFT'):
                np.save(file=full_name + ".npy", arr=adv_image)
                if args.save_out and result.original_class_id == args.True_class_id:
                    adv_images.append(adv_image)
                    adv_labels.append(result.adv_class_id)
                    org_images.append(original_image)
                    org_labels.append(result.original_class_id)
                    gauss_images.append(gauss_image)
                    gauss_labels.append(result.gauss_class_id)
                    if result.gauss_class_id == result.original_class_id:
                        adv_recovered_images.append(adv_image)
                        adv_recovered_labels.append(result.adv_class_id)
                        org_recovered_images.append(original_image)
                        org_recovered_labels.append(result.original_class_id)
                        gauss_recovered_images.append(gauss_image)
                        gauss_recovered_labels.append(result.gauss_class_id)

        if show_diff:
            # omit the diff image in the spatial domain.
            args.plot_index += col_diff

        if show_2nd:
            pass
            # omit the diff image in the spatial domain.
            # args.plot_index += 2

        result.image_index = args.image_index
        # write labels to the log file.
        with open(args.file_name_labels, "a") as f:
            if args.total_count == 0:
                header = result.get_attrs_sorted(delimiter=delimiter)
                f.write(header + "\n")
            values = result.get_vals_sorted(delimiter=delimiter)
            f.write(values + "\n")

        for fft_type in fft_types:
            args.fft_type = fft_type
            for channel in channels:
                is_log = True
                if args.show_original:
                    image_fft = print_fft(image=original_image,
                                          channel=channel,
                                          args=args,
                                          title="frequency domain",
                                          # title="original",
                                          is_log=is_log)

                if show_2nd and original_image2 is not None:
                    image2_fft = print_fft(image=original_image2,
                                           channel=channel,
                                           args=args,
                                           # title="original 2nd",
                                           title="target",
                                           is_log=is_log)

                if adv_image is not None and args.adv_type == AdversarialType.BEFORE:
                    adversarial_fft = print_fft(image=adv_image,
                                                channel=channel,
                                                args=args,
                                                title="adversarial",
                                                is_log=is_log)

                if args.values_per_channel > 0:
                    rounded_fft = print_fft(image=rounded_image,
                                            channel=channel,
                                            args=args,
                                            title="cd (color depth)",
                                            is_log=is_log)

                if args.compress_fft_layer > 0:
                    compressed_fft = print_fft(image=compress_image,
                                               channel=channel,
                                               args=args,
                                               title="fc (fft compression)",
                                               is_log=is_log)
                if args.noise_sigma > 0:
                    gauss_fft = print_fft(image=gauss_image,
                                          channel=channel,
                                          args=args,
                                          title="gaussian noise",
                                          is_log=is_log)

                if args.noise_epsilon > 0:
                    noise_fft = print_fft(image=noise_image,
                                          channel=channel,
                                          args=args,
                                          title="uniform noise",
                                          is_log=is_log)

                if args.laplace_epsilon > 0:
                    laplace_fft = print_fft(image=noise_image,
                                            channel=channel,
                                            args=args,
                                            title="laplace noise",
                                            is_log=is_log)

                if args.svd_compress > 0:
                    svd_fft = print_fft(image=svd_image,
                                        channel=channel,
                                        args=args,
                                        title="svd",
                                        is_log=is_log)

                if adv_image is not None and args.attack_type == "after":
                    adversarial_fft = print_fft(image=adv_image,
                                                channel=channel,
                                                args=args,
                                                title="adversarial",
                                                is_log=is_log)

                if show_diff:
                    diff_fft = adversarial_fft / image_fft
                    ylabel = "fft-ed channel " + str(channel) + ":\n" + fft_type
                    diff_fft_avg = np.average(diff_fft)
                    print_heat_map(diff_fft, args=args,
                                   title="fft(adv) / fft(original)\n"
                                   f"(avg: {diff_fft_avg})",
                                   ylabel=ylabel)

                    diff_fft = adversarial_fft - image_fft
                    diff_fft_avg = np.average(diff_fft)
                    ylabel = "fft-ed channel " + str(channel) + ":\n" + fft_type
                    print_heat_map(diff_fft, args=args,
                                   title="fft(adv) - fft(original)\n"
                                   f"(avg: {diff_fft_avg})",
                                   ylabel=ylabel)

                if show_2nd:
                    if show_2nd_col_diff:
                        diff_fft = image2_fft / image_fft
                        ylabel = "fft-ed channel " + str(
                            channel) + ":\n" + fft_type
                        diff_fft_avg = np.average(diff_fft)
                        print_heat_map(diff_fft, args=args,
                                       title="fft(original2) / fft(original)\n"
                                       f"(avg: {diff_fft_avg})",
                                       ylabel=ylabel)

                        diff_fft = image2_fft - image_fft
                        diff_fft_avg = np.average(diff_fft)
                        ylabel = "fft-ed channel " + str(
                            channel) + ":\n" + fft_type
                        print_heat_map(diff_fft, args=args,
                                       title="fft(original2) - fft(original)\n"
                                       f"(avg: {diff_fft_avg})",
                                       ylabel=ylabel)

        plt.subplots_adjust(hspace=0.0, left=0.075, right=0.9)

    format = 'png'  # "pdf" or "png" file_name
    attack_name = "None"
    if attack != None:
        attack_name = attack.name()
    file_name = "images2/" + attack_name + "-" + str(
        args.attack_type) + "-round-fft-" + str(
        args.compress_fft_layer) + "-" + args.dataset + "-channel-" + str(
        channels_nr) + "-" + "val-per-channel-" + str(
        args.values_per_channel) + "-noise-epsilon-" + str(
        args.noise_epsilon) + "-noise-sigma-" + str(
        args.noise_sigma) + "-laplace-epsilon-" + str(
        args.compress_rate) + "-compress-rate-" + str(
        args.laplace_epsilon) + "-img-idx-" + str(
        args.image_index) + "-" + get_log_time()

    if args.is_debug:
        pass
        print("file name: ", file_name)
        plt.savefig(fname=file_name + "." + format, format=format)
        plt.show(block=True)
    plt.close()
    return result


def randomized_defense(image, fmodel, original_image=None, defense_name=""):
    """
    the randomized defense.

    :param image: the adversarial image
    :param fmodel: the foolbox "wrapped" model
    :param original_image: the original image

    :return: the result of the defense: recovered label, l2 distance, etc. in
    the result object.
    """
    if args.recover_iterations > 0 or args.noise_iterations > 0:
        # number of random trials and classifications to select the
        # final recovered class based on the plurality: the input image
        # is perturbed many times with random noise, we record class
        # for each trial and return as the result the class that was
        # selected most times.
        # noise_iterations is also used in the attack.
        # recover_iterations is used only for the defense.
        # if args.noise_iterations > 0 and args.recover_iterations > 0:
        #     raise Exception(f"only one iterations for recovery can be "
        #                     "set but "
        #                     "noise_iterations={args.noise_iterations} "
        #                     "and "
        #                     "recover_iterations={args.recover_iterations}.")
        if args.recover_iterations > 0:
            iters = args.recover_iterations
        elif args.noise_iterations > 0:
            iters = args.noise_iterations
        result_noise, _, _ = defend(
            image=image,
            fmodel=fmodel,
            args=args,
            iters=iters,
            original_image=original_image)
        print(
            f"{defense_name} recovered label, id by {args.noise_iterations} "
            f"iterations: ",
            result_noise.label, result_noise.class_id)
        return result_noise
    return Object()


def index_ranges(
        input_ranges=[(20, 58), (2500, 2516), (5000, 5050), (9967, 10000)]):
    """
    generate list of indexes with the given pair of inclusive ranges.

    :param ranges:
    :return:
    """
    indexes = np.array([], dtype=np.int)
    for range_start, range_end in input_ranges:
        indexes = np.concatenate(
            (indexes, [x for x in range(range_start, range_end + 1)]))
    return indexes


def result_file(args):
    args.file_name_labels = results_folder + args.recover_type + "-" + args.interpolate + "-round-fft-" + str(
        args.compress_fft_layer) + "-" + args.dataset + "-" + "val-per-channel-" + str(
        args.values_per_channel) + "-noise-epsilon-" + str(
        args.noise_epsilon) + "-noise-sigma-" + str(
        args.noise_sigma) + "-laplace-epsilon-" + str(
        args.laplace_epsilon) + "-noise-iterations-" + str(
        args.noise_iterations) + "-" + get_log_time()
    with open(args.file_name_labels, "a") as f:
        f.write(args.get_str() + "\n\n")


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    # save fft representations of the original and adversarial images to files
    args.save_out = True
    # args.diff_type = "source"  # "source" or "fft"
    args.diff_type = "fft"
    args.show_original = True
    args.global_log_time = get_log_time()
    # args.max_iterations = 1000
    # args.max_iterations = 20
    # args.noise_iterations = 1
    # args.dataset = "cifar10"  # "cifar10" or "imagenet"
    # args.dataset = "imagenet"
    # args.dataset = "mnist"
    # args.index = 13  # index of the image (out of 20) to be used
    # args.compress_rate = 0
    # args.interpolate = "exp"
    index_range = range(args.start_epoch, args.epochs, args.step_size)
    # index_range = range(11, 12)
    # index_range = range(249, 250)
    # index_range = range(0, 1000)
    # index_range = [10000]
    # index_range = range(60, 100)
    # index_range = range(0, 100, 100)
    # index_range = range(82, 83)
    # index_range = range(263, 264)
    # index_range = range(296, 297)
    # index_range = range(0, 1000)
    # index_range = [1000]
    if args.is_debug:
        args.use_foolbox_data = False
    if not args.use_foolbox_data:
        train_loader, test_loader, train_dataset, test_dataset, limit = get_data(
            args=args)

    # args.compress_fft_layer = 5
    # args.values_per_channel = 8

    if torch.cuda.is_available() and args.use_cuda:
        print("cuda is available")
        args.device = torch.device("cuda")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("cuda id not available")
        args.device = torch.device("cpu")

    if args.recover_type == "rounding":
        # val_range = [2, 4, 8, 16, 32, 64, 128, 256]
        val_range = args.many_values_per_channel
        if args.is_debug:
            val_range = [8]
        # val_range = range(261, 1, -5)
        # val_range = range(200, 261, 5)
        # val_range = range(260, 200, -5)
    elif args.recover_type == "fft":
        # val_range = [1, 10, 20, 30, 40, 50, 60, 80, 90]
        val_range = args.compress_rates
        # val_range = range(1, 100, 2)
        # val_range = range(3)
    elif args.recover_type == "roundfft":
        val_range = range(1)
    elif args.recover_type == "fftround":
        val_range = range(1)
    elif args.recover_type == "fftuniform":
        val_range = range(1)
    elif args.recover_type == "roundsvd":
        val_range = range(1)
    elif args.recover_type == "rounduniform":
        val_range = range(1)
    elif args.recover_type == "gauss":
        # val_range = [0.001, 0.009, 0.03, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]
        val_range = args.noise_sigmas
        # val_range = [0.03]
        # if args.is_debug:
        #     val_range = [0.009]
        # val_range = [x / 1000 for x in range(10)]
        # val_range += [x / 100 for x in range(1, 51)]
        # val_range += [x / 100 for x in range(1, 11)]
        # val_range += [x / 100 for x in range(11, 31)]
        # val_range += [x / 100 for x in range(31, 51)]
        # val_range += [x / 100 for x in range(51, 0, -1)]
    elif args.recover_type == "noise":
        val_range = args.noise_epsilons
        if args.is_debug:
            val_range = [0.03]
    elif args.recover_type == "laplace":
        val_range = args.laplace_epsilons
        if args.is_debug:
            val_range = [0.03]
    elif args.recover_type == "svd":
        val_range = args.many_svd_compress
    elif args.recover_type == "debug":
        val_range = [0.009]
    elif args.recover_type == "all":
        val_range = [0.0]
    elif args.recover_type == "empty":
        val_range = [0.0]
    else:
        raise Exception(f"Unknown recover type: {args.recover_type}")

    print(args.get_str())
    out_recovered_file = results_folder + "out_" + args.recover_type + "_recovered-" + str(
        args.dataset) + "-" + str(
        args.values_per_channel) + "-" + str(
        args.compress_fft_layer) + "-" + "-noise-epsilon-" + str(
        args.noise_epsilon) + "-noise-sigma-" + str(
        args.noise_sigma) + "-laplace_epsilon-" + str(
        args.laplace_epsilon) + "-interpolate-" + str(
        args.interpolate) + "-" + str(
        args.attack_name) + "-" + str(
        args.attack_type) + "-" + get_log_time() + ".txt"
    with open(out_recovered_file, "a") as f:
        f.write(args.get_str() + "\n")
        header = ["compress_" + args.recover_type + "_layer",
                  "total_count",
                  "recover iterations",
                  "noise iterations",
                  "attack strength",
                  "noise epsilon",
                  "laplace epsilon",
                  "svd compress",
                  "attack max iterations",
                  "% base accuracy",
                  "% total accuracy after attack",
                  "% of adversarials",
                  "% total accuracy after recovery",
                  "% of recovered from adversarial",
                  "% total accuracy after many recovery",
                  "% of many recovered from adversarial",
                  "avg. L2 distance defense",
                  "avg. L1 distance defense",
                  "avg. Linf distance defense",
                  "avg. confidence defense",
                  "avg. L2 distance defense many",
                  "avg. L1 distance defense many",
                  "avg. Linf distance defense many",
                  "avg. confidence defense many",
                  "avg. L2 distance attack",
                  "avg. L1 distance attack",
                  "avg. Linf distance attack",
                  "avg. confidence attack",
                  "# of recovered",
                  "# of recovered many",
                  "count adv",
                  "adv time (sec)",
                  "defend time (sec)",
                  "run time (sec)\n"]
        f.write(delimiter.join(header))

    result_file(args)
    # for compress_fft_layer in [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 45, 50, 60, 75, 80, 90, 99]:
    for compress_value in val_range:
        print("compress_" + args.recover_type + "_layer: ", compress_value)
        if args.recover_type == "debug":
            args.values_per_channel = 0
            args.compress_fft_layer = 50
            args.compress_rate = 50
            args.noise_sigma = 0.0
            args.noise_epsilon = 0.0
            args.laplace_epsilon = 0.0
        elif args.recover_type == "fft":
            args.compress_fft_layer = compress_value
            args.compress_rate = compress_value
        elif args.recover_type == "rounding":
            args.values_per_channel = compress_value
        elif args.recover_type == "gauss":
            args.noise_sigma = compress_value
        elif args.recover_type == "noise":
            args.noise_epsilon = compress_value
        elif args.recover_type == "laplace":
            args.laplace_epsilon = compress_value
        elif args.recover_type == "svd":
            args.svd_compress = compress_value
        elif args.recover_type == "roundfft":
            pass
        elif args.recover_type == "fftround":
            pass
        elif args.recover_type == "fftuniform":
            pass
        elif args.recover_type == "roundsvd":
            pass
        elif args.recover_type == "rounduniform":
            pass
        elif args.recover_type == "all" or args.recover_type == "empty":
            pass
        else:
            raise Exception(
                f"Unknown recover type: {args.recover_type}")

        # recover_iterations = [0]  # + [2 ** x for x in range(1, 9)]
        # iterations = [0]
        # if lenargs.many_recover_iterations > 0 and args.many_noise_iterations > 0:
        #     raise Exception('Set either many_recover_iterations or '
        #                     'many_noise_iterations in args.')
        # if args.many_attack_iterations > 0 and args.many_noise_iterations > 0:
        #     raise Exception('Set either many_attack_iterations or '
        #                     'many_noise_iterations in args.')
        # if args.many_recover_iterations > 0:
        #     iterations = args.many_recover_iterations
        # elif args.many_attack_iterations
        for recover_iter in args.many_recover_iterations:
            args.recover_iterations = recover_iter
            for noise_iter in args.many_noise_iterations:
                args.noise_iterations = noise_iter
                for attack_iterations in args.many_attack_iterations:
                    args.attack_max_iterations = attack_iterations
                    for attack_strength in args.attack_strengths:
                        args.attack_strength = attack_strength

                        count_original = 0  # how many examples correctly classified by the base model
                        count_recovered = 0
                        count_many_recovered = 0
                        count_adv = 0  # count the number of adversarial examples that have label different than the ground truth
                        count_incorrect = 0
                        total_adv = 0  # adversarial might return the correct label as the transformation can cause mis-prediction
                        sum_L1_distance_defense = 0
                        sum_L2_distance_defense = 0
                        sum_Linf_distance_defense = 0
                        sum_confidence_defense = 0
                        sum_L1_distance_defense_many = 0
                        sum_L2_distance_defense_many = 0
                        sum_Linf_distance_defense_many = 0
                        sum_confidence_defense_many = 0
                        sum_L2_distance_adv = 0
                        sum_L1_distance_adv = 0
                        sum_Linf_distance_adv = 0
                        sum_confidence_adv = 0
                        sum_adv_timing = 0
                        sum_defend_timing = 0
                        args.total_count = 0

                        # for index in range(4950, -1, -50):
                        # for index in range(0, 5000, 50):
                        # for index in range(0, 20):

                        run_time = 0

                        for image_index in index_range:
                            # for index in range(args.start_epoch, limit, step):
                            # for index in range(args.start_epoch, 5000, 50):
                            # for index in range(limit - step, args.start_epoch - 1, -step):
                            args.image_index = image_index
                            print("image index: ", image_index)

                            start = time.time()

                            result_run = run(args)

                            single_run_time = time.time() - start

                            args.total_count += 1
                            print("single run elapsed time: ", single_run_time)
                            run_time += single_run_time

                            if result_run.original_label is not None and (
                                    result_run.true_label == result_run.original_label):
                                count_original += 1

                            if args.recover_type == "rounding":
                                if result_run.round_label is not None:
                                    if result_run.true_label == result_run.round_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.round_L2_distance
                                    sum_L1_distance_defense += result_run.round_L1_distance
                                    sum_Linf_distance_defense += result_run.round_Linf_distance
                                    sum_confidence_defense += result_run.round_confidence

                            elif args.recover_type == "fft":
                                if result_run.fft_label is not None:
                                    if result_run.true_label == result_run.fft_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.fft_L2_distance
                                    sum_L1_distance_defense += result_run.fft_L1_distance
                                    sum_Linf_distance_defense += result_run.fft_Linf_distance
                                    sum_confidence_defense += result_run.fft_confidence

                            elif args.recover_type == "roundfft":
                                if result_run.roundfft_label is not None:
                                    if result_run.true_label == result_run.roundfft_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.roundfft_L2_distance
                                    sum_L1_distance_defense += result_run.roundfft_L1_distance
                                    sum_Linf_distance_defense += result_run.roundfft_Linf_distance
                                    sum_confidence_defense += result_run.roundfft_confidence

                            elif args.recover_type == "roundsvd":
                                if result_run.roundsvd_label is not None:
                                    if result_run.true_label == result_run.roundsvd_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.roundsvd_L2_distance
                                    sum_L1_distance_defense += result_run.roundsvd_L1_distance
                                    sum_Linf_distance_defense += result_run.roundsvd_Linf_distance
                                    sum_confidence_defense += result_run.roundsvd_confidence

                            elif args.recover_type == "rounduniform":
                                if result_run.rounduniform_label is not None:
                                    if result_run.true_label == result_run.rounduniform_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.rounduniform_L2_distance
                                    sum_L1_distance_defense += result_run.rounduniform_L1_distance
                                    sum_Linf_distance_defense += result_run.rounduniform_Linf_distance
                                    sum_confidence_defense += result_run.rounduniform_confidence

                            elif args.recover_type == "fftround":
                                if result_run.fftround_label is not None:
                                    if result_run.true_label == result_run.fftround_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.fftround_L2_distance
                                    sum_L1_distance_defense += result_run.fftround_L1_distance
                                    sum_Linf_distance_defense += result_run.fftround_Linf_distance
                                    sum_confidence_defense += result_run.fftround_confidence

                            elif args.recover_type == "fftuniform":
                                if result_run.fftuniform_label is not None:
                                    if result_run.true_label == result_run.fftuniform_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.fftuniform_L2_distance
                                    sum_L1_distance_defense += result_run.fftuniform_L1_distance
                                    sum_Linf_distance_defense += result_run.fftuniform_Linf_distance
                                    sum_confidence_defense += result_run.fftuniform_confidence

                            elif args.recover_type == "gauss":
                                if result_run.gauss_label is not None:
                                    if result_run.true_label == result_run.gauss_label:
                                        count_recovered += 1
                                sum_L2_distance_defense += result_run.gauss_L2_distance
                                sum_L1_distance_defense += result_run.gauss_L1_distance
                                sum_Linf_distance_defense += result_run.gauss_Linf_distance
                                sum_confidence_defense += result_run.gauss_confidence
                                if args.noise_iterations > 0 or args.recover_iterations > 0:
                                    if result_run.true_label == result_run.gauss_many_label:
                                        count_many_recovered += 1
                                    sum_L2_distance_defense_many += result_run.gauss_many_L2_distance
                                    sum_L1_distance_defense_many += result_run.gauss_many_L1_distance
                                    sum_Linf_distance_defense_many += result_run.gauss_many_Linf_distance
                                    sum_confidence_defense_many += result_run.gauss_many_confidence
                                    sum_defend_timing += result_run.time_gauss_defend

                            elif args.recover_type == "noise":
                                if result_run.noise_label is not None:
                                    if result_run.true_label == result_run.noise_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.noise_L2_distance
                                    sum_L1_distance_defense += result_run.noise_L1_distance
                                    sum_Linf_distance_defense += result_run.noise_Linf_distance
                                    sum_confidence_defense += result_run.noise_confidence
                                if args.noise_iterations > 0 or args.recover_iterations > 0:
                                    if result_run.true_label == result_run.noise_many_label:
                                        count_many_recovered += 1
                                    sum_L2_distance_defense_many += result_run.noise_many_L2_distance
                                    sum_L1_distance_defense_many += result_run.noise_many_L1_distance
                                    sum_Linf_distance_defense_many += result_run.noise_many_Linf_distance
                                    sum_confidence_defense_many += result_run.noise_many_confidence
                                    sum_defend_timing += result_run.time_noise_defend

                            elif args.recover_type == "laplace":
                                if result_run.laplace_label is not None:
                                    if result_run.true_label == result_run.laplace_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.laplace_L2_distance
                                    sum_L1_distance_defense += result_run.laplace_L1_distance
                                    sum_Linf_distance_defense += result_run.laplace_Linf_distance
                                    sum_confidence_defense += result_run.laplace_confidence
                                if args.noise_iterations > 0 or args.recover_iterations > 0:
                                    if result_run.true_label == result_run.laplace_many_label:
                                        count_many_recovered += 1
                                    sum_L2_distance_defense_many += result_run.laplace_many_L2_distance
                                    sum_L1_distance_defense_many += result_run.laplace_many_L1_distance
                                    sum_Linf_distance_defense_many += result_run.laplace_many_Linf_distance
                                    sum_confidence_defense_many += result_run.laplace_many_confidence
                                    sum_defend_timing += result_run.time_laplace_defend

                            elif args.recover_type == "svd":
                                if result_run.svd_label is not None:
                                    if result_run.true_label == result_run.svd_label:
                                        count_recovered += 1
                                    sum_L2_distance_defense += result_run.svd_L2_distance
                                    sum_L1_distance_defense += result_run.svd_L1_distance
                                    sum_Linf_distance_defense += result_run.svd_Linf_distance
                                    sum_confidence_defense += result_run.svd_confidence
                                if args.noise_iterations > 0 or args.recover_iterations > 0:
                                    if result_run.true_label == result_run.svd_many_label:
                                        count_many_recovered += 1
                                    sum_L2_distance_defense_many += result_run.svd_many_L2_distance
                                    sum_L1_distance_defense_many += result_run.svd_many_L1_distance
                                    sum_Linf_distance_defense_many += result_run.svd_many_Linf_distance
                                    sum_confidence_defense_many += result_run.svd_many_confidence
                                    sum_defend_timing += result_run.time_svd_defend
                            elif args.recover_type == "debug":
                                exit(0)
                            elif args.recover_type == "all" or args.recover_type == "empty":
                                pass
                            else:
                                raise Exception(
                                    f"Unknown recover type: {args.recover_type}")

                            if result_run.original_label is not None and (
                                    result_run.true_label == result_run.original_label):
                                # The classifier was correct.
                                if result_run.adv_label is not None:
                                    if result_run.original_label != result_run.adv_label:
                                        # The classifier was fooled.
                                        count_adv += 1
                                        # Aggregate the statistics about the attack.
                                        sum_L2_distance_adv += result_run.adv_L2_distance
                                        sum_L1_distance_adv += result_run.adv_L1_distance
                                        sum_Linf_distance_adv += result_run.adv_Linf_distance
                                        sum_confidence_adv += result_run.adv_confidence
                                        sum_adv_timing += result_run.adv_timing

                            if result_run.true_label != result_run.original_label or (
                                    result_run.adv_label is not None and result_run.original_label != result_run.adv_label):
                                count_incorrect += 1

                            if image_index % 128 == 0:
                                save_adv_org()

                        total_count = args.total_count

                        print("total_count: ", total_count)
                        print("classified correctly (base model): ",
                              count_original)
                        print("adversarials found: ", count_adv)
                        print("recovered correctly: ", count_recovered)

                        if count_adv > 0:
                            avg_L2_distance_adv = sum_L2_distance_adv / count_adv
                            avg_L1_distance_adv = sum_L1_distance_adv / count_adv
                            avg_Linf_distance_adv = sum_Linf_distance_adv / count_adv
                            avg_confidence_adv = sum_confidence_adv / count_adv
                            percent_of_recovered_from_adversarials = count_recovered / count_adv * 100
                            percent_of_many_recovered_from_adversarials = count_many_recovered / count_adv * 100
                        else:
                            avg_L2_distance_adv = 0.0
                            avg_L1_distance_adv = 0.0
                            avg_Linf_distance_adv = 0.0
                            avg_confidence_adv = 0.0
                            percent_of_recovered_from_adversarials = 0.0
                            percent_of_many_recovered_from_adversarials = 0.0

                        base_accuracy = count_original / total_count * 100
                        accuracy_after_attack = (
                                                        1 - count_incorrect / total_count) * 100
                        if count_original > 0:
                            percent_of_adversarial_from_correctly_classified = count_adv / count_original * 100
                        else:
                            percent_of_adversarial_from_correctly_classified = 0
                            print(
                                'The image was misclassified                            percent_of_adversarial_from_correctly_classified!')
                        recovered_accuracy = count_recovered / total_count * 100
                        recovered_many_accuracy = count_many_recovered / total_count * 100

                        with open(out_recovered_file, "a") as f:
                            f.write(delimiter.join([str(x) for x in
                                                    [compress_value,
                                                     total_count,
                                                     args.recover_iterations,
                                                     args.noise_iterations,
                                                     args.attack_strength,
                                                     args.noise_epsilon,
                                                     args.laplace_epsilon,
                                                     args.svd_compress,
                                                     args.attack_max_iterations,
                                                     base_accuracy,
                                                     accuracy_after_attack,
                                                     percent_of_adversarial_from_correctly_classified,
                                                     recovered_accuracy,
                                                     percent_of_recovered_from_adversarials,
                                                     recovered_many_accuracy,
                                                     percent_of_many_recovered_from_adversarials,
                                                     sum_L2_distance_defense / total_count,
                                                     sum_L1_distance_defense / total_count,
                                                     sum_Linf_distance_defense / total_count,
                                                     sum_confidence_defense / total_count,
                                                     sum_L2_distance_defense_many / total_count,
                                                     sum_L1_distance_defense_many / total_count,
                                                     sum_Linf_distance_defense_many / total_count,
                                                     sum_confidence_defense_many / total_count,
                                                     avg_L2_distance_adv,
                                                     avg_L1_distance_adv,
                                                     avg_Linf_distance_adv,
                                                     avg_confidence_adv,
                                                     count_recovered,
                                                     count_many_recovered,
                                                     count_adv,
                                                     sum_adv_timing,
                                                     sum_defend_timing,
                                                     run_time]]) + "\n")

    save_adv_org()

    print("total elapsed time: ", time.time() - start_time)
