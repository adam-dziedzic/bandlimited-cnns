import torchvision.models as models

from cnns.nnlib.datasets.imagenet.imagenet_from_class_idx_to_label import \
    imagenet_from_class_idx_to_label
from cnns.nnlib.datasets.imagenet.imagenet_from_class_label_to_idx import \
    imagenet_from_class_label_to_idx
from cnns.nnlib.datasets.cifar10_from_class_idx_to_label import \
    cifar10_from_class_idx_to_label
from cnns.nnlib.datasets.cifar10_from_class_label_to_idx import \
    cifar10_from_class_label_to_idx

from cnns.nnlib.utils.model_utils import load_model
from cnns.nnlib.datasets.cifar import cifar_max, cifar_min
from cnns.nnlib.datasets.cifar import cifar_std_array, cifar_mean_array
from cnns.nnlib.datasets.cifar import cifar_mean_mean
from cnns.nnlib.datasets.mnist.mnist import mnist_max, mnist_min
from cnns.nnlib.datasets.mnist.mnist import mnist_mean_mean
from cnns.nnlib.datasets.mnist.mnist import mnist_std_array, mnist_mean_array
from cnns.nnlib.datasets.mnist.mnist_from_class_idx_to_label import \
    mnist_from_class_idx_to_label
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_max
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_min
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_mean_array
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_std_array
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_mean_mean


def get_model(args):
    if args.dataset == "imagenet":
        args.cmap = None
        args.init_y, args.init_x = 224, 224

        min = imagenet_min
        max = imagenet_max
        args.min = min
        args.max = max
        args.mean_array = imagenet_mean_array
        args.mean_mean = imagenet_mean_mean
        args.std_array = imagenet_std_array
        args.num_classes = 1000
        pytorch_model = models.resnet50(pretrained=True).eval().to(
            device=args.device)
        # network_model = load_model(args=args, pretrained=True)
        args.from_class_idx_to_label = imagenet_from_class_idx_to_label

    elif args.dataset == "cifar10":
        args.cmap = None
        args.init_y, args.init_x = 32, 32
        args.num_classes = 10
        # args.values_per_channel = 0
        # args.model_path = "saved_model_2019-04-08-16-51-16-845688-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.22-channel-vals-0.model"
        # args.model_path = "saved_model_2019-04-08-19-41-48-571492-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.5.model"
        # args.model_path = "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"
        # args.model_path = "saved_model_2019-04-13-06-54-15-810999-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-91.64-channel-vals-8.model"
        # args.compress_rate = 5
        # args.compress_rates = [args.compress_rate]
        # if args.model_path == "no_model":
        # args.model_path = "saved-model-2019-05-11-22-20-59-242197-dataset-cifar10-preserve-energy-100-compress-rate-5.0-test-accuracy-93.43-channel-vals-0.model"
        # args.attack_type = AttackType.BAND_ONLY
        # args.model_path = "saved_model_2019-04-13-10-25-49-315784-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-accuracy-93.08-channel-vals-32.model"
        # args.model_path = "saved_model2019-05-11-18-54-18-392325-dataset-cifar10-preserve-energy-100.0-compress-rate-5.0-test-accuracy-91.21-channel-vals-8.model"
        args.in_channels = 3
        min = cifar_min
        max = cifar_max
        args.min = min
        args.max = max
        args.mean_array = cifar_mean_array
        args.mean_mean = cifar_mean_mean
        args.std_array = cifar_std_array
        pytorch_model = load_model(args=args)
        args.from_class_idx_to_label = cifar10_from_class_idx_to_label

    elif args.dataset == "mnist":
        args.cmap = "gray"
        args.init_y, args.init_x = 28, 28
        args.num_classes = 10
        # args.values_per_channel = 2
        # args.model_path = "2019-05-03-10-08-51-149612-dataset-mnist-preserve-energy-100-compress-rate-0.0-test-accuracy-99.07-channel-vals-0.model"
        # args.compress_rate = 0
        # args.compress_rates = [args.compress_rate]
        args.in_channels = 1
        min = mnist_min
        max = mnist_max
        args.min = min
        args.max = max
        args.mean_array = mnist_mean_array
        args.mean_mean = mnist_mean_mean
        args.std_array = mnist_std_array
        # args.network_type = NetworkType.Net
        pytorch_model = load_model(args=args)
        args.from_class_idx_to_label = mnist_from_class_idx_to_label

    else:
        raise Exception(f"Unknown dataset type: {args.dataset}")

    return pytorch_model