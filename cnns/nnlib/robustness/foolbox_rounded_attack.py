"""
Check how many images are classified incorrectly because of the rounding attack.
"""

from cnns import matplotlib_backend

print("Using:", matplotlib_backend.backend)
import torch
from cnns.nnlib.utils.exec_args import get_args
import numpy as np
import time
from cnns.nnlib.robustness.utils import get_foolbox_model
from cnns.nnlib.robustness.utils import Rounder
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_min
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_max
import foolbox
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet
import torchvision.models as models


def run(args):
    if args.dataset == "cifar10":
        train_loader, test_loader, train_dataset, test_dataset = get_cifar(args,
                                                                           args.dataset)
        model_path = "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"
        compress_rate = 0
        fmodel = get_foolbox_model(args, model_path=model_path,
                                   compress_rate=compress_rate)
    elif args.dataset == "imagenet":
        train_loader, test_loader, train_dataset, test_dataset = load_imagenet(args)
        # model = models.resnet18(pretrained=True).eval()
        model = models.resnet50(pretrained=True).eval()
        model.to(device=args.device)
        fmodel = foolbox.models.PyTorchModel(model, bounds=(
            imagenet_min, imagenet_max), num_classes=args.num_classes)

    for spacing in [2 ** x for x in range(0, 8)]:
        start_time = time.time()
        incorrect = 0
        counter = 0
        rounder = Rounder(spacing)
        for batch_idx, (data, target) in enumerate(test_loader):
            # print("batch_idx: ", batch_idx)
            for i, label in enumerate(target):
                counter += 1
                label = label.item()
                image = data[i].numpy()

                predictions = fmodel.predictions(image)
                # The image has to be classified correctly in the first place.
                if np.argmax(predictions) == label:
                    counter += 1

                    # round the adversarial image
                    image = rounder.round(image)

                    predictions = fmodel.predictions(image)
                    # print(np.argmax(predictions), label)
                    if np.argmax(predictions) != label:
                        incorrect += 1
        timing = time.time() - start_time
        with open(args.out_file_name, "a") as out:
            msg = ",".join((str(x) for x in
                            [spacing, incorrect, counter,
                             incorrect / counter * 100.0,
                             timing, rounder.get_average_diff_per_pixel()]))
            print(msg)
            out.write(msg + "\n")


if __name__ == "__main__":
    np.random.seed(31)
    # arguments
    args = get_args()
    # customized arguments
    # args.dataset = "imagenet"  # "cifar10" or "imagenet"
    args.sample_count_limit = 1000

    if torch.cuda.is_available() and args.use_cuda:
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda id not available")
        args.device = torch.device("cpu")

    print("dataset: ", args.dataset)
    header = "spacing, incorrect, counter, incorrect rate (%), time (sec), " \
             "avg diff per pixel"
    print(header)
    args.out_file_name = "rounding_attack_" + args.dataset + ".csv"
    with open(args.out_file_name, "a") as out:
        out.write(header + "\n")

    run(args)
