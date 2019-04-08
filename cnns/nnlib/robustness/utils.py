import numpy as np
from cnns.nnlib.datasets.cifar10_example import cifar10_example
import foolbox
import torch
import os
from cnns.nnlib.pytorch_architecture.resnet2d import resnet18
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.cifar import cifar_mean_array
from cnns.nnlib.datasets.cifar import cifar_std_array
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet

# min and max values per pixel in cifar10 test data
cifar_min = -2.4290658
cifar_max = 2.7537313


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


def load_model(args):
    model = resnet18(args=args)
    # load pretrained weights
    models_folder_name = "models"
    models_dir = os.path.join(os.getcwd(), os.path.pardir,
                              "pytorch_experiments", models_folder_name)
    if args.model_path != "no_model":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=args.device))
        msg = "loaded model: " + args.model_path
        # logger.info(msg)
        # print(msg)
    return model.eval()


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
                 values_per_channel=255,
                 mean=cifar_mean_array,
                 std=cifar_std_array):
        self.round_multiplier = values_per_channel
        self.ext_multiplier = 1.0 / self.round_multiplier
        self.mean = np.array(mean, dtype=np.float32).reshape((3, 1, 1))
        self.std = np.array(std, dtype=np.float32).reshape((3, 1, 1))
        self.sum_diff = 0.0
        self.count_diffs = 0

    def round(self, image):
        image = unnormalize(image, self.mean, self.std)

        # print("image max min: ", np.max(image), np.min(image))

        # round the image
        round_image = self.ext_multiplier * np.round(
            self.round_multiplier * image)

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
