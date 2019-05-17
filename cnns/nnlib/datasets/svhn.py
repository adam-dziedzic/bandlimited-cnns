import torchvision
from torchvision import transforms
import torch
from cnns.nnlib.utils.general_utils import MemoryType
import numpy as np
from cnns.nnlib.utils.general_utils import NetworkType

svhn_mean = (0.4914, 0.4822, 0.4465)
svhn_std = (0.2023, 0.1994, 0.2010)

svhn_mean_array = np.array(svhn_mean, dtype=np.float32).reshape((3, 1, 1))
svhn_std_array = np.array(svhn_std, dtype=np.float32).reshape((3, 1, 1))

# -0.42421296 2.8214867 -0.42421296 2.8214867 False False
# svhn_min = np.float(-0.42421296)
# svhn_max = np.float(2.8214867)
# min:  -2.4290657  max:  2.7537313
svhn_min = np.float32(-2.43)
svhn_max = np.float32(2.76)

def get_svhn(args):
    """"
    Get SVHN data.

    :param args: the general arguments for a program, e.g. memory type of debug
    mode.
    :return: main train and test loaders, as well as other params, such as
    number of classes.
    """
    args.num_classes = 10
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
    args.mean_array = svhn_mean_array
    args.std_array = svhn_std_array
    if args.network_type is NetworkType.LE_NET:
        args.out_channels = [10, 20]
        args.signal_dimension = 2
    elif args.network_type is NetworkType.ResNet18:
        args.signal_dimension = 2
    elif args.network_type is NetworkType.ResNet50:
        args.signal_dimension = 2
    elif args.network_type is NetworkType.DenseNetCifar:
        args.signal_dimension = 2
    else:
        raise Exception(f"Uknown network type: {args.network_type.name}")

    train_dataset = torchvision.datasets.SVHN(
        root='./data', split='train',
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(svhn_mean, svhn_std)
            ])
    )
    if sample_count > 0:
        train_dataset.train_data = train_dataset.train_data[:sample_count]
        train_dataset.train_labels = train_dataset.train_labels[
                                     :sample_count]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.min_batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_dataset = torchvision.datasets.SVHN(
        root='./data', split="test",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(svhn_mean, svhn_std)
            ])
    )
    if sample_count > 0:
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
    args.sample_count_limit = 0
    train_loader, test_loader, train_dataset, test_dataset = get_svhn(args)
    min = np.float("inf")
    max = np.float("-inf")
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("batch_idx: ", batch_idx)
        for i, label in enumerate(target):
            label = label.item()
            image = data[i].numpy()
            min_image = np.min(image)
            max_image = np.max(image)
            # print(i, min_image, max_image)
            if min_image < min:
                min = min_image
            if max_image > max:
                max = max_image
    print("min: ", min, " max: ", max)
    # show_images()
