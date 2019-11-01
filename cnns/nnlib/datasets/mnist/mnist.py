import torchvision
from torchvision import transforms
import torch
from cnns.nnlib.utils.general_utils import MemoryType
import numpy as np
from cnns.nnlib.datasets.transformations.svd_compression import \
    SVDCompressionTransformation

mnist_mean = (0.1307,)
mnist_mean_mean = 0.1307
mnist_std = (0.3081,)

mnist_mean_array = np.array(mnist_mean, dtype=np.float32).reshape((1, 1, 1))
mnist_std_array = np.array(mnist_std, dtype=np.float32).reshape((1, 1, 1))

# -0.42421296 2.8214867 -0.42421296 2.8214867 False False
# mnist_min = np.float(-0.42421296)
# mnist_max = np.float(2.8214867)
mnist_min = np.float32(-0.425)
mnist_max = np.float32(2.8215)

def get_transform(args):
    transformations = []
    transformations.append(transforms.ToTensor())
    transformations.append(transforms.Normalize(mnist_mean, mnist_std))
    if args and args.svd_transform > 0.0:
        transformations.append(
            SVDCompressionTransformation(args=args,
                                         compress_rate=args.svd_transform)
        )
    transform_train = transforms.Compose(transformations)
    return transform_train

def get_mnist(args):
    """
    Get the MNIST dataset.

    :param args: the general arguments for a program, e.g. memory type of debug
    mode.
    :return: main train and test loaders, as well as other params, such as
    number of classes.
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
    args.input_height = 28
    args.input_width = 28
    args.width = args.input_height * args.input_width
    args.in_channels = 1  # number of channels in the input data
    args.out_channels = [10, 20]

    args.min = mnist_min
    args.max = mnist_max
    args.mean_array = mnist_mean_array
    args.mean_mean = mnist_mean_mean
    args.std_array = mnist_std_array

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True,
        download=True,
        transform=get_transform(args=args)
    )
    if sample_count > 0:
        train_dataset.train_data = train_dataset.train_data[:sample_count]
        train_dataset.train_labels = train_dataset.train_labels[
                                     :sample_count]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.min_batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False,
        download=True,
        transform=get_transform(args=args)
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
    train_loader, test_loader = get_mnist(args)
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
