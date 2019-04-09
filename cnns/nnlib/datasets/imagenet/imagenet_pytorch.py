import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from cnns.nnlib.utils.exec_args import get_args
import numpy as np

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

imagenet_mean_array = np.array(imagenet_mean).reshape((3, 1, 1))
imagenet_std_array = np.array(imagenet_std).reshape((3, 1, 1))

# the min/max value per pixel after normalization
imagenet_min = -2.2 # -2.1179039478302
imagenet_max = 2.7  # 2.640000104904175

def load_imagenet(args):
    args.num_classes = 1000
    traindir = os.path.join(args.imagenet_path, 'train')
    valdir = os.path.join(args.imagenet_path, 'val')
    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    sample_count = args.sample_count_limit
    if sample_count > 0:
        train_dataset.imgs = train_dataset.imgs[:sample_count]

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.min_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    if sample_count > 0:
        val_dataset.imgs = val_dataset.imgs[:sample_count]

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.min_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    args = get_args()
    args.dataset = "imagenet"
    load_imagenet(args)
