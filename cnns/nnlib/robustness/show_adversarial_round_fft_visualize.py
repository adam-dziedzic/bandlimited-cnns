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
# from cnns import matplotlib_backend
# print("Using:", matplotlib_backend.backend)
import matplotlib

import time
import matplotlib.pyplot as plt
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
from cnns.nnlib.datasets.transformations.normalize import Normalize
from cnns.nnlib.datasets.transformations.denormalize import Denormalize
from cnns.nnlib.utils.general_utils import NetworkType
from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet
from cnns.nnlib.utils.complex_mask import get_hyper_mask
from cnns.nnlib.pytorch_layers.pytorch_utils import MockContext
from cnns.nnlib.utils.shift_DC_component import shift_to_center

def softmax(x):
    s = np.exp(x - np.max(x))
    s /= np.sum(s)
    return s


# def softmax(x):
#     s = torch.nn.functional.softmax(torch.tensor(x, dtype=torch.float))
#     return s.numpy()


def to_fft_type(xfft, fft_type, is_log=True):
    if fft_type == "magnitude":
        return to_fft_magnitude(xfft, is_log)
    elif fft_type == "phase":
        return to_fft_phase(xfft)
    else:
        raise Exception(f"Unknown type of fft processing: {fft_type}")


def to_fft(x, fft_type, is_log=True, onesided=False, to_center=False):
    x = torch.from_numpy(x)
    # x = torch.tensor(x)
    # x = x.permute(2, 0, 1)  # move channel as the first dimension
    xfft = torch.rfft(x, onesided=onesided, signal_ndim=2)
    if to_center:
        xfft = shift_to_center(xfft, onesided=onesided)
    return to_fft_type(xfft, fft_type=fft_type, is_log=is_log)


def to_xfft(x, fft_type, args, is_log=True, channel=0, onesided=False,
            to_center=False):
    """

    :param x: input image
    :param fft_type: magnitude or phase
    :param args: arguments of the program
    :param is_log:
    :param channel:
    :return:
    """
    xfft = to_fft(x, fft_type, is_log=is_log, onesided=onesided,
                  to_center=to_center)
    xfft = xfft[channel, :args.init_x, :args.init_y]
    return xfft


def from_ctx_fft(xfft, fft_type, is_log=True, channel=0):
    xfft = to_fft_type(xfft, fft_type=fft_type, is_log=is_log)
    xfft = xfft[channel, :args.init_x, :args.init_y]
    return xfft


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
        args.values_per_channel = 0
        from_class_idx_to_label = imagenet_from_class_idx_to_label

    elif args.dataset == "cifar10":
        args.cmap = None
        args.init_y, args.init_x = 32, 32
        args.num_classes = 10
        args.values_per_channel = 0
        # args.model_path = "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"
        args.model_path = "saved_model_2019-04-13-06-54-15-810999-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-91.64-channel-vals-8.model"
        args.compress_rate = 0
        args.compress_rates = [args.compress_rate]
        args.in_channels = 3
        min = cifar_min
        max = cifar_max
        args.min = min
        args.max = max
        args.mean_array = cifar_mean_array
        args.std_array = cifar_std_array
        args.network_type = NetworkType.ResNet18
        network_model = load_model(args=args)
        from_class_idx_to_label = cifar10_from_class_idx_to_label
    elif args.dataset == "mnist":
        args.cmap = "gray"
        args.init_y, args.init_x = 28, 28
        args.num_classes = 10
        args.values_per_channel = 0
        args.model_path = "2019-05-03-10-08-51-149612-dataset-mnist-preserve-energy-100-compress-rate-0.0-test-accuracy-99.07-channel-vals-0.model"
        args.compress_rate = 0
        args.compress_rates = [args.compress_rate]
        args.in_channels = 1
        min = mnist_min
        max = mnist_max
        args.min = min
        args.max = max
        args.mean_array = mnist_mean_array
        args.std_array = mnist_std_array
        args.network_type = NetworkType.Net
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

    matplotlib.rcParams.update({'font.size': 18})

    cmap_type = "matshow"  # "standard" or "custom"
    # cmap_type = "standard"

    # vmin_heatmap = -6
    # vmax_heatmap = 10

    vmin_heatmap = None
    vmax_heatmap = None

    decimals = 3

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

    def print_heat_map(input_map, args=args, title="", ylabel="", vmin=None,
                       vmax=None):
        args.plot_index += 1
        plt.subplot(rows, cols, args.plot_index)
        plt.ylabel(ylabel)
        plt.title(title)

        # zero out the center
        # H, W = input_map.shape
        # input_map[H // 2, W // 2] = 0
        # np.set_printoptions(threshold=np.inf)
        # print("xfft: ", input_map)
        print("xfft min: ", input_map.min())
        print("xfft max: ", input_map.max())

        interpolation = "nearest"
        plt.imshow(input_map, cmap='hot', interpolation=interpolation,
                   vmin=vmin, vmax=vmax)
        # plt.contour(input_map, cmap='hot', interpolation='nearest',
        #            vmin=vmin, vmax=vmax)
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
    attacks = [
        CarliniWagnerL2AttackRoundFFT(model=fmodel, args=args)
        # foolbox.attacks.CarliniWagnerL2Attack(fmodel),
        # foolbox.attacks.FGSM(fmodel),
        # foolbox.attacks.AdditiveUniformNoiseAttack(fmodel)
    ]
    # 1 is for the first row of images.
    rows = 3
    cols = 2

    fig, big_axes = plt.subplots(figsize=(8, rows * 4), nrows=rows, ncols=cols,
                                 sharey=True)

    # for row, big_ax in enumerate(big_axes, start=1):
    #     big_ax.set_title("Subplot row %s \n" % row, fontsize=16)
    #
    #     # Turn off axis lines and ticks of the big subplot
    #     # obs alpha is 0 in RGBA string!
    #     big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off',
    #                        bottom='off', left='off', right='off')
    #     # removes the white frame
    #     big_ax._frameon = False

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
    else:
        train_loader, test_loader, train_dataset, test_dataset = load_imagenet(
            args)

    for attack in attacks:
        def show_image(image, original_image, title="", ylabel="", args=args):
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
            # title_str = title + '\n'
            # title_str += 'label: ' + str(
            #     predicted_label.replace(",", "\n")) + "\n"
            confidence_str = str(np.around(confidence, decimals=decimals))
            info_str = "confidence: " + confidence_str + ", "
            L2_distance = meter.measure(original_image, image)
            L2_distance_str = str(np.around(L2_distance, decimals=decimals))
            if args.plot_index > 0:
                info_str += "L2 distance: " + L2_distance_str + "\n"
            print("info_str: ", info_str)
            image_show = denormalizer.denormalize(image)
            if args.dataset == "mnist":
                # image_show = image_show.astype('uint8')
                # plt.imshow(image_show.squeeze(), cmap=args.cmap,
                #            interpolation='nearest')
                print_heat_map(input_map=image_show[0], args=args,
                               ylabel=ylabel, title=title)
            else:
                args.plot_index += 1
                plt.subplot(rows, cols, args.plot_index)
                plt.title(title)
                plt.imshow(
                    np.moveaxis(image, 0, -1),
                    # move channels to last dimension
                    cmap=args.cmap)
                # plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
                # plt.axis('off')
                # if args.plot_index % cols == 1:
                plt.ylabel(ylabel)

            return predicted_label, confidence, L2_distance

        fft_type = "magnitude"

        def print_fft(image, channel, title="", ylabel="", args=args):
            print("fft: ", title)
            print("input min: ", image.min())
            print("input max: ", image.max())
            xfft = to_fft(image, fft_type=fft_type)
            xfft = xfft[channel, ...]
            xfft = xfft[:lim_y, :lim_x]
            torch.set_printoptions(profile='full')
            print("original_fft size: ", xfft.shape)
            options = np.get_printoptions()
            np.set_printoptions(threshold=np.inf)
            print("xfft min: ", xfft.min())
            print("xfft max: ", xfft.max())
            torch.set_printoptions(profile='default')
            # go back to the original print size
            np.set_printoptions(threshold=options['threshold'])

            # print("original_fft:\n", original_fft)
            # print_color_map(xfft, fig, ax)
            print_heat_map(xfft, args=args, title=title, ylabel=ylabel)

            return xfft

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

        channel = 0
        is_log = True
        onesided = False
        is_clipped = True
        shift_DC_to_center = True

        original_xfft = to_xfft(original_image, fft_type=fft_type,
                                args=args, channel=channel, is_log=is_log,
                                onesided=onesided,
                                to_center=shift_DC_to_center)

        ctx = MockContext()
        complex_compress_image = attack.fft_complex_compression(
            original_image, get_mask=get_hyper_mask, ctx=ctx, onesided=onesided)
        if is_clipped:
            complex_xfft = to_xfft(complex_compress_image, fft_type=fft_type,
                                   args=args, channel=channel, is_log=is_log,
                                   onesided=onesided,
                                   to_center=shift_DC_to_center)
        else:
            complex_xfft = from_ctx_fft(ctx.xfft[0], is_log=True,
                                        fft_type=fft_type,
                                        channel=channel)


        lshaped_compress_image = attack.fft_lshape_compression(
            original_image, ctx=ctx, onesided=onesided)
        if is_clipped:
            lshaped_xfft = to_xfft(lshaped_compress_image, fft_type=fft_type,
                                   args=args, channel=channel, is_log=is_log,
                                   onesided=onesided,
                                   to_center=shift_DC_to_center)
        else:
            lshaped_xfft = from_ctx_fft(ctx.xfft[0], is_log=True,
                                        fft_type=fft_type,
                                        channel=channel)

        min_val_fft = np.min([original_xfft.min(),
                              complex_xfft.min(),
                              lshaped_xfft.min()])
        # This is to make the visualization pretty, otherwise the FFT changes
        # are not that visible.
        min_val_fft = original_xfft.min() - 33
        print("min val: ", min_val_fft)

        max_val_fft = np.max([original_xfft.max(),
                              complex_xfft.max(),
                              lshaped_xfft.max()])
        print("max val: ", max_val_fft)

        title_left = "Spatial domain"
        title_right = "FFT domain"

        # Original image.
        show_image(image=original_image, original_image=original_image,
                   title=title_left, ylabel="Original image")

        # print_fft(image=original_image, channel=channel, title="FFT domain")
        print_heat_map(original_xfft, args=args, title=title_right,
                       vmin=min_val_fft, vmax=max_val_fft)

        # fft const disk
        show_image(image=complex_compress_image,
                   original_image=original_image,
                   title="Spatial domain",
                   ylabel="Exact Circular\ncompression")
        # print_fft(image=complex_compress_image, channel=channel,
        #           title="FFT domain")
        print_heat_map(complex_xfft, args=args, title=title_right,
                       vmin=min_val_fft, vmax=max_val_fft)

        # fft lshape

        # title = "compressed image (" + str(args.compress_fft_layer) + "%)"
        show_image(image=lshaped_compress_image,
                   original_image=original_image,
                   title="Spatial domain",
                   ylabel="Approximate L-shaped\ncompression")

        # print_fft(image=lshaped_compress_image, channel=channel,
        #           title="FFT domain")
        print_heat_map(lshaped_xfft, args=args, title=title_right,
                       vmin=min_val_fft, vmax=max_val_fft)

        plt.subplots_adjust(hspace=0.6)

    format = 'pdf'  # "pdf" or "png"
    file_name = "images/" + "visualize-fft-compression-" + args.dataset + "-channel-" + str(
        channels_nr) + "-" + "img-idx-" + str(
        args.index) + "-" + get_log_time()
    print("file name: ", file_name)
    plt.savefig(fname=file_name + "." + format, format=format)
    plt.show(block=True)
    plt.close()


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    # args.diff_type = "source"  # "source" or "fft"
    args.diff_type = "fft"
    # args.dataset = "cifar10"  # "cifar10" or "imagenet"
    args.dataset = "imagenet"
    # args.dataset = "mnist"
    # args.index = 13  # index of the image (out of 20) to be used
    args.compress_rate = 0
    args.compress_fft_layer = 50
    args.is_fft_compression = True
    args.interpolate = "const"
    args.use_foolbox_data = True
    args.index = 0

    if torch.cuda.is_available() and args.use_cuda:
        print("cuda is available")
        args.device = torch.device("cuda")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("cuda id not available")
        args.device = torch.device("cpu")

    run(args)

    print("total elapsed time: ", time.time() - start_time)
