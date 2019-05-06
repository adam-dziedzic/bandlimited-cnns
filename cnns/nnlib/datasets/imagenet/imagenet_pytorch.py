import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import numpy as np
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.datasets.transformations.dtype_transformation import \
    DtypeTransformation
from cnns.nnlib.datasets.transformations.flat_transformation import \
    FlatTransformation
from cnns.nnlib.datasets.transformations.gaussian_noise import \
    AddGaussianNoiseTransformation
from cnns.nnlib.datasets.transformations.rounding import RoundingTransformation

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

imagenet_mean_array = np.array(imagenet_mean, dtype=np.float32).reshape(
    (3, 1, 1))
imagenet_std_array = np.array(imagenet_std, dtype=np.float32).reshape((3, 1, 1))

# the min/max value per pixel after normalization
imagenet_min = np.float(-2.1179039478302)  # -2.1179039478302
imagenet_max = np.float(2.640000104904175)  # 2.640000104904175

normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)


def get_transform_train(args, dtype=torch.float32, signal_dimension=2):
    transformations = []
    transformations.append(transforms.RandomResizedCrop(224))
    transformations.append(transforms.RandomHorizontalFlip())
    transformations.append(transforms.ToTensor())
    # if args.values_per_channel > 0:
    #     transformations.append(
    #         RoundingTransformation(values_per_channel=args.values_per_channel))
    transformations.append(normalize)
    if signal_dimension == 1:
        transformations.append(FlatTransformation())
    # transformations.append(DtypeTransformation(dtype=dtype))
    transform_train = transforms.Compose(transformations)
    return transform_train


def get_transform_test(args, dtype=torch.float32, signal_dimension=2,
                       noise_sigma=0):
    transformations = []
    transformations.append(transforms.Resize(256))
    transformations.append(transforms.CenterCrop(224))
    transformations.append(transforms.ToTensor())
    transformations.append(normalize)
    # if args.values_per_channel > 0:
    #     transformations.append(
    #         RoundingTransformation(values_per_channel=args.values_per_channel))
    if signal_dimension == 1:
        transformations.append(FlatTransformation())
    if noise_sigma > 0:
        transformations.append(
            AddGaussianNoiseTransformation(sigma=noise_sigma))
    # transformations.append(DtypeTransformation(dtype))
    transform_test = transforms.Compose(transformations)
    return transform_test


def load_imagenet(args):
    args.num_classes = 1000
    pin_memory = False
    if args.memory_type is MemoryType.PINNED:
        pin_memory = True
    args.in_channels = 3  # number of channels in the input data
    args.out_channels = None
    args.signal_dimension = 2
    args.mean_array = imagenet_mean_array
    args.std_array = imagenet_std_array

    traindir = os.path.join(args.imagenet_path, 'train')
    valdir = os.path.join(args.imagenet_path, 'val')

    train_dataset = datasets.ImageFolder(
        traindir, get_transform_train(args=args))

    sample_count = args.sample_count_limit
    if sample_count > 0:
        train_dataset.imgs = train_dataset.imgs[:sample_count]
        train_dataset.samples = train_dataset.samples[:sample_count]
        train_dataset.classes = train_dataset.classes[:sample_count]

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        train_sampler.num_samples = sample_count
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.min_batch_size,
        shuffle=(train_sampler is None),
        # has to be False if train_sampler provided
        num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler)
    # train_loader.batch_sampler.sampler.num_samples = sample_count

    val_dataset = datasets.ImageFolder(valdir, get_transform_test(args=args))

    if sample_count > 0:
        val_dataset.imgs = val_dataset.imgs[:sample_count]
        val_dataset.samples = val_dataset.samples[:sample_count]
        val_dataset.classes = val_dataset.classes[:sample_count]

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.min_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory)

    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    from cnns.nnlib.utils.exec_args import get_args

    args = get_args()
    args.dataset = "imagenet"
    train_loader, test_loader, train_dataset, test_dataset = load_imagenet(args)
    counter = 0
    while True:
        test_dataset.__getitem__(counter)
        counter +=1
        print("counter: ", counter)
