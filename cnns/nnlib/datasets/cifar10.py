from torchvision import transforms
import torch
from cnns.nnlib.datasets.transformations.dtype_transformation import DtypeTransformation
from cnns.nnlib.datasets.transformations.flat_transformation import FlatTransformation

def get_transform_train(dtype=torch.float32, signal_dimension=2):
    transformations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]
    if signal_dimension == 1:
        transformations.append(FlatTransformation())
    transformations.append(DtypeTransformation(dtype=dtype))
    transform_train = transforms.Compose(transformations)
    return transform_train


def get_transform_test(dtype=torch.float32, signal_dimension=2):
    transformations = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]
    if signal_dimension == 1:
        transformations.append(FlatTransformation())
    transformations.append(DtypeTransformation(dtype))
    transform_test = transforms.Compose(transformations)
    return transform_test