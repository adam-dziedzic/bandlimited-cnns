from torchvision import transforms
import torch
from cnns.nnlib.utils.general_utils import MemoryType
import numpy as np
from cnns.nnlib.datasets.synthetic.dataset import SyntheticDataset
from cnns.nnlib.datasets.transformations.dtype_transformation import \
    DtypeTransformation
from cnns.nnlib.datasets.transformations.flat_transformation import \
    FlatTransformation
from cnns.nnlib.datasets.transformations.svd_compression import \
    SVDCompressionTransformation


def get_transform_train(args):
    transformations = []
    if args and args.svd_transform > 0.0:
        transformations.append(
            SVDCompressionTransformation(args=args,
                                         compress_rate=args.svd_transform)
        )
    transform_train = transforms.Compose(transformations)
    return transform_train


def get_transform_test(args):
    transformations = []
    if args and args.svd_transform > 0.0:
        transformations.append(
            SVDCompressionTransformation(args=args,
                                         compress_rate=args.svd_transform)
        )
    transform_test = transforms.Compose(transformations)
    return transform_test


def get_synthetic(args):
    """
    Get the synthetic dataset.
    """
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
    args.num_classes = 10
    args.flat_size = 320  # the size of the flat vector after the conv layers in LeNet
    args.width = 28 * 28
    args.in_channels = 1  # number of channels in the input data
    args.out_channels = [10, 20]

    train_dataset = SyntheticDataset(train=True, num_classes=args.num_classes)
    if sample_count > 0:
        train_dataset.set_length(length=sample_count)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.min_batch_size,
        shuffle=True,
        transform=get_transform_train(
            args=args,
            dtype=torch.float,
            signal_dimension=args.signal_dimension),
        **kwargs)

    test_dataset = SyntheticDataset(train=False, num_classes=args.num_classes)
    if sample_count > 0:
        test_dataset.set_length(length=sample_count)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        transform=get_transform_test(
            args=args,
            dtype=torch.float,
            signal_dimension=args.signal_dimension,
            noise_sigma=args.noise_sigma),
        **kwargs)

    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == "__main__":
    from cnns.nnlib.utils.exec_args import get_args

    args = get_args()
    args.sample_count_limit = 100
    train_loader, test_loader, train_dataset, test_dataset = get_synthetic(
        args=args)
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
