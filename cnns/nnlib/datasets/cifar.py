from torchvision import transforms
from torchvision import datasets
import torch
from cnns.nnlib.datasets.transformations.dtype_transformation import \
    DtypeTransformation
from cnns.nnlib.datasets.transformations.flat_transformation import \
    FlatTransformation
from cnns.nnlib.datasets.transformations.gaussian_noise import \
    AddGaussianNoiseTransformation
from cnns.nnlib.datasets.transformations.rounding import RoundingTransformation
from cnns.nnlib.utils.general_utils import MemoryType
from cnns.nnlib.utils.general_utils import NetworkType
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle
import os


print("current directory is: ", os.getcwd())

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2023, 0.1994, 0.2010)

cifar_mean_array = np.array(cifar_mean, dtype=np.float32).reshape((3, 1, 1))
cifar_std_array = np.array(cifar_std, dtype=np.float32).reshape((3, 1, 1))

# the min/max value per pixel after normalization
# exact values:
# counter:  10000  min:  -2.429065704345703  max:  2.7537312507629395
cifar_min = -2.5  # -2.429065704345703
cifar_max = 2.8  # 2.7537312507629395


def show_images():
    """
    Prints 5X5 grid of random Cifar10 images. It isn't blurry, though not perfect either.
    https://stackoverflow.com/questions/35995999/why-cifar-10-images-are-not-displayed-properly-using-matplotlib
    """
    f = open('./data/cifar-10-batches-py/data_batch_1', 'rb')
    datadict = cPickle.load(f, encoding='latin1')
    f.close()
    X = datadict["data"]
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y = np.array(Y)

    # Visualizing CIFAR 10
    fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            # X[i:i+1] keeps the value in a table, so simply use X[i].
            axes1[j][k].imshow(X[i])
    plt.show()


def get_transform_train(args, dtype=torch.float32, signal_dimension=2):
    transformations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    # if args.values_per_channel > 0:
    #     transformations.append(
    #         RoundingTransformation(values_per_channel=args.values_per_channel))
    transformations.append(transforms.Normalize(cifar_mean, cifar_std))
    if signal_dimension == 1:
        transformations.append(FlatTransformation())
    transformations.append(DtypeTransformation(dtype=dtype))
    transform_train = transforms.Compose(transformations)
    return transform_train


def get_transform_test(args, dtype=torch.float32, signal_dimension=2,
                       noise_sigma=0):
    transformations = [transforms.ToTensor()]
    # if args.values_per_channel > 0:
    #     transformations.append(
    #         RoundingTransformation(values_per_channel=args.values_per_channel))
    transformations.append(transforms.Normalize(cifar_mean, cifar_std))
    if signal_dimension == 1:
        transformations.append(FlatTransformation())
    if noise_sigma > 0:
        transformations.append(
            AddGaussianNoiseTransformation(sigma=noise_sigma))
    transformations.append(DtypeTransformation(dtype))
    transform_test = transforms.Compose(transformations)
    return transform_test


def get_cifar(args, dataset_name):
    """
    Get the MNIST dataset.

    :param args: the general arguments for a program, e.g. memory type of debug
    mode.
    :return: main train and test loaders, as well as other params, such as
    number of classes.
    """
    if dataset_name == "cifar10":
        args.num_classes = 10
        dataset_loader = datasets.CIFAR10
    elif dataset_name == "cifar100":
        args.num_classes = 100
        dataset_loader = datasets.CIFAR100
    else:
        raise Exception(f"Uknown dataset: {dataset_name}")

    sample_count = args.sample_count_limit
    use_cuda = args.use_cuda
    num_workers = args.workers
    pin_memory = False
    if args.memory_type is MemoryType.PINNED:
        pin_memory = True
    if use_cuda:
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    else:
        kwargs = {'num_workers': num_workers}
    args.width = 32 * 32
    # The size of the flat vector after the conv layers in LeNet.
    args.flat_size = 500
    args.in_channels = 3  # number of channels in the input data
    args.out_channels = None
    args.signal_dimension = 2
    if args.network_type is NetworkType.LE_NET:
        args.out_channels = [10, 20]
        args.signal_dimension = 2
    elif args.network_type is NetworkType.ResNet18:
        args.signal_dimension = 2
    elif args.network_type is NetworkType.DenseNetCifar:
        args.signal_dimension = 2
    else:
        raise Exception(f"Uknown network type: {args.network_type.name}")
    train_dataset = dataset_loader(root='./data', train=True,
                                   download=True,
                                   transform=get_transform_train(
                                       args=args,
                                       dtype=torch.float,
                                       signal_dimension=args.signal_dimension))
    if sample_count > 0:
        try:
            train_dataset.data = train_dataset.data[:sample_count]
            train_dataset.targets = train_dataset.targets[:sample_count]
        except AttributeError:
            train_dataset.train_data = train_dataset.train_data[:sample_count]
            train_dataset.train_labels = train_dataset.train_labels[
                                         :sample_count]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.min_batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_dataset = dataset_loader(root='./data', train=False,
                                  download=True,
                                  transform=get_transform_test(
                                      args=args,
                                      dtype=torch.float,
                                      signal_dimension=args.signal_dimension,
                                      noise_sigma=args.noise_sigma))
    if sample_count > 0:
        try:
            test_dataset.data = test_dataset.data[:sample_count]
            test_dataset.targets = test_dataset.targets[:sample_count]
        except AttributeError:
            test_dataset.test_data = test_dataset.test_data[:sample_count]
            test_dataset.test_labels = test_dataset.test_labels[:sample_count]

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              **kwargs)

    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == "__main__":
    from cnns.nnlib.utils.exec_args import get_args
    args = get_args()
    args.sample_count_limit = 3
    train_loader, test_loader, train_dataset, test_dataset = get_cifar(args,
                                                                       "cifar10")
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("batch_idx: ", batch_idx)
        for i, label in enumerate(target):
            label = label.item()
            image = data[i].numpy()
            print(i, np.max(image), np.min(image))
    # show_images()
