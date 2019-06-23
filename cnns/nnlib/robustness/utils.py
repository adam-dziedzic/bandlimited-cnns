import numpy as np
from numpy.testing.utils import assert_equal
from cnns.nnlib.datasets.cifar10_example import cifar10_example
import foolbox
import torch
import os
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.cifar import cifar_mean_array
from cnns.nnlib.datasets.cifar import cifar_std_array
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet
from cnns.nnlib.datasets.cifar import cifar_min
from cnns.nnlib.datasets.cifar import cifar_max
from cnns.nnlib.datasets.transformations.rounding import RoundingTransformation

from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from cnns.nnlib.pytorch_layers.pytorch_utils import get_phase
from cnns.nnlib.utils.shift_DC_component import shift_DC
from foolbox.attacks.additive_noise import AdditiveNoiseAttack

nprng = np.random.RandomState()


def softmax(logits):
    s = np.exp(logits - np.max(logits))
    s /= np.sum(s)
    return s


def most_frequent_class(predictions):
    # 1. Get the max prediction from each random noise (row).
    max_each_row = np.argmax(predictions, axis=1)
    # 2. Get the most frequent class.
    predicted_class_id = np.bincount(max_each_row).argmax()
    return predicted_class_id


def softmax_from_torch(x):
    s = torch.nn.functional.softmax(torch.tensor(x, dtype=torch.float))
    return s.numpy()


def uniform_noise(epsilon, shape, dtype, args):
    """
    Similar to foolbox but batched version.
    :param epsilon: strength of the noise
    :param bounds: min max for images
    :param shape: the output shape
    :param dtype: the output type
    :return: the noise for images
    """
    w = epsilon * (args.max - args.min)
    noise = nprng.uniform(-w, w, size=shape)
    noise = noise.astype(dtype)
    return noise


def gauss_noise(epsilon, shape, dtype, args):
    """
    Similar to foolbox but batched version.
    :param epsilon: strength of the noise
    :param bounds: min max for images
    :param shape: the output shape
    :param dtype: the output type
    :return: the noise for images
    """
    std = epsilon / np.sqrt(3) * (args.max - args.min)
    noise = nprng.normal(scale=std, size=shape)
    noise = noise.astype(dtype)
    return noise


class AdditiveLaplaceNoiseAttack(AdditiveNoiseAttack):
    """Adds uniform noise to the image, gradually increasing
    the standard deviation until the image is misclassified.

    """
    def __init__(self, args):
        super(AdditiveLaplaceNoiseAttack, self).__init__()
        self.args = args

    def _sample_noise(self, epsilon, image, bounds):
        return laplace_noise(epsilon=epsilon, shape=image.shape,
                             dtype=image.dtype, args=self.args)

def laplace_noise(epsilon, shape, dtype, args):
    """
    Similar to foolbox but batched version.
    :param epsilon: strength of the noise
    :param bounds: min max for images
    :param shape: the output shape
    :param dtype: the output type
    :return: the noise for images
    """
    scale = epsilon / np.sqrt(3) * (args.max - args.min)
    noise = nprng.laplace(loc=args.mean_mean, scale=scale, size=shape)
    noise = noise.astype(dtype)
    return noise


def norm(x, p=2, axis=(1, 2, 3)):
    """
    This is a batch computation of the norms. We calculate the norm along
    for all axis in axis. Numpy and PyTorch support only a single dimension for
    such computation. The 2 dimensional tuple is for singular values.

    :param x: the input tensor
    :param p: p-norm
    :param axis: axis for norm calculation for each tensor in the batch, this
    are essentially the dimensions of each tensor, excluding the batch dimension.
    :return: the p-norms for each tensor in the batch
    """
    if p == 0:
        return np.sum(x != 0, axis=axis)
    elif p == 1:
        return np.sum(np.abs(x), axis=axis)
    elif p == float("inf") or p == "inf":
        return np.max(np.abs(x), axis=axis)
    elif p is None or p == 2:
        # special case for speedup
        return np.sqrt(np.sum(np.square(x), axis=axis))
    else:
        return np.power(np.sum(np.power(x, p), axis=axis), 1.0 / p)


def elem_wise_dist(image, images, p=2, axis=(1, 2, 3)):
    """
    Element wise p-norm dist along many axis.

    :param image: a single image (np.array)
    :param images: many images (np.array)
    :param p: the p-norm (1,2, or inf)
    :return: average distance between every image from images and the first image

    >>> t1 = np.array([[[1.0,2,3]]])
    >>> t2_1 = [[[2.0,2,3]]]
    >>> t2_2 = [[[1.0, 4,3]]]
    >>> t2 = np.array([t2_1, t2_2])
    >>> dist_all = elem_wise_dist(t1, t2)
    >>> # print("dist all: ", dist_all)
    >>> assert_equal(actual=dist_all, desired=np.array([1, 2]))
    """
    x = image - images  # broadcast image to as many instances as in images
    return norm(x, p=p, axis=axis)


def to_fft(x, fft_type, is_log=True, signal_dim=2, onesided=False,
           is_DC_shift=False):
    x = torch.from_numpy(x)
    # x = torch.tensor(x)
    # x = x.permute(2, 0, 1)  # move channel as the first dimension
    xfft = torch.rfft(x, onesided=onesided, signal_ndim=signal_dim)
    if is_DC_shift:
        xfft = shift_DC(xfft, onesided=onesided)
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
        # xfft = np.clip(xfft, 1e-12, None)

        # Shift tensor +1: zeros become ones, but after log, they are zeros
        # again.
        xfft += 1

        # min_xfft = xfft.min()
        # print("min xfft: ", min_xfft)
        xfft = 20 * np.log10(xfft)  # Decibel scale.
        # xfft = np.log10(xfft) / np.log10(1000000)

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


def normalize(x, mean, std):
    _mean = mean.astype(x.dtype)
    _std = std.astype(x.dtype)
    result = x - _mean
    result /= _std
    return result


def unnormalize(x, mean=cifar_mean_array, std=cifar_std_array):
    """
    >>> x = np.array(cifar10_example)
    >>> mean = np.array(cifar_mean, dtype=np.float32).reshape((3, 1, 1))
    >>> std = np.array(cifar_std, dtype=np.float32).reshape((3, 1, 1))
    >>> x = unnormalize(x, mean, std)
    >>> max, min = np.max(x), np.min(x)
    >>> assert max <= 1.0, f"max is: {max}"
    >>> assert min >= 0.0, f"min is: {min}"

    :param x: input normalized image
    :param mean: the mean array of shape (3,1,1)
    :param std: the std array of shape (3,1,1)
    :return: the unnormalized tensor
    """
    _mean = mean.astype(x.dtype)
    _std = std.astype(x.dtype)
    result = x * std
    result += mean
    result = np.clip(result, 0.0, 1.0)
    return result

def get_foolbox_model(args, model_path, compress_rate=0, min=cifar_min,
                      max=cifar_max):
    """
    :param args: arguments for the whole program
    :param model_path: the path to the pytorch model to be loaded
    :param compress_rate: the compression rate of the loaded model
    :param min: min value per image for cifar10 test data
    :param max: max value per image for cifar10 test data
    :return: the foolbox model
    """
    args.model_path = model_path
    args.compress_rate = compress_rate
    args.compress_rates = [compress_rate]
    pytorch_model = load_model(args=args)
    # do not change the mean and standard deviation when they are 0 and 1
    # we already normalized the data via the pytorch torchvision
    mean = 0
    std = 1
    foolbox_model = foolbox.models.PyTorchModel(model=pytorch_model,
                                                bounds=(min, max),
                                                num_classes=args.num_classes,
                                                preprocessing=(mean, std),
                                                device=args.device)
    return foolbox_model


def get_min_max_counter_cifar10():
    args = get_args()
    # should we turn pixels to the range from 0 to 255 and round them to the the
    # nearest integer value
    args.sample_count_limit = 0
    train_loader, test_loader, train_dataset, test_dataset = get_cifar(args,
                                                                       "cifar10")
    min, max, counter = get_min_max_counter(test_loader)
    return min, max, counter


def get_min_max_counter_imagenet():
    args = get_args()
    # should we turn pixels to the range from 0 to 255 and round them to the the
    # nearest integer value
    args.sample_count_limit = 100
    train_loader, test_loader, train_dataset, test_dataset = load_imagenet(args)
    min, max, counter = get_min_max_counter(test_loader)
    return min, max, counter


def get_min_max_counter(test_loader):
    """
    :param test_loader: test data
    :return: min, max values in each image, and the counter of images
    """
    min = float("inf")
    max = float("-inf")

    counter = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        # print("batch_idx: ", batch_idx)
        for i, label in enumerate(target):
            counter += 1
            image = data[i]
            if image.min().item() < min:
                min = image.min().item()
            if image.max().item() > max:
                max = image.max().item()
    # print("counter: ", counter, " min: ", min, " max: ", max)
    return min, max, counter


class Rounder():
    """
    Round the values of the pixels. From 256 values per color channel to fewer
    values, e.g. 128 values per channel.
    """

    def __init__(self,
                 values_per_channel=256,
                 mean=cifar_mean_array,
                 std=cifar_std_array):
        self.rounder = RoundingTransformation(
            values_per_channel=values_per_channel, rounder=np.round)
        self.mean = np.array(mean, dtype=np.float32).reshape((3, 1, 1))
        self.std = np.array(std, dtype=np.float32).reshape((3, 1, 1))
        self.sum_diff = 0.0
        self.count_diffs = 0

    def round(self, image):
        image = unnormalize(image, self.mean, self.std)

        # print("image max min: ", np.max(image), np.min(image))

        # round the image
        round_image = self.rounder(image)

        # stats
        diff = np.abs(round_image - image)
        self.sum_diff += np.sum(diff) / (diff.size * 1.0)
        self.count_diffs += 1

        image = normalize(round_image, self.mean, self.std)
        return image

    def get_average_diff_per_pixel(self):
        if self.count_diffs == 0:
            return np.inf
        return self.sum_diff / self.count_diffs


if __name__ == "__main__":
    print("run utils examples")
    # min, max, counter = get_min_max_counter_cifar10()
    min, max, counter = get_min_max_counter_imagenet()
    print("counter: ", counter, " min: ", min, " max: ", max)
