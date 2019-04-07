"""
Import only the adversarial examples.
"""

from cnns import matplotlib_backend

print("Using:", matplotlib_backend.backend)
import torch
from cnns.nnlib.utils.exec_args import get_args
import numpy as np
import time
from cnns.nnlib.robustness.utils import get_foolbox_model
from cnns.nnlib.robustness.utils import Rounder
import tables
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_min
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_max
import foolbox
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import load_imagenet
import torchvision.models as models

def run(args):
    if args.dataset == "cifar10":
        get_cifar(args, args.dataset)
        model_path = "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"
        compress_rate = 0
        fmodel = get_foolbox_model(args, model_path=model_path,
                                   compress_rate=compress_rate)
    elif args.dataset == "imagenet":
        load_imagenet(args)
        # model = models.resnet18(pretrained=True).eval()
        model = models.resnet50(pretrained=True).eval()
        model.to(device=args.device)
        fmodel = foolbox.models.PyTorchModel(model, bounds=(
            imagenet_min, imagenet_max), num_classes=1000)

    # input data
    # filename = 'outarray.h5'  # test example
    # filename = 'outarray_adverse'
    # filename = 'outarray_adverse_200'
    filename = args.dataset + '_outarray_adverse_2_10000'
    f = tables.open_file(filename + ".h5", mode='r')
    max_advers_imgs = len(f.root.data)
    # max_advers_imgs = 10
    # labels = [x for x in range(max_advers_imgs)]
    labels = np.load(filename + '.labels.npy')
    for spacing in [2 ** x for x in range(0, 8)]:
        start_time = time.time()
        correct = 0
        counter = 0
        rounder = Rounder(spacing)
        for i in range(max_advers_imgs):
            label = labels[i]
            image = f.root.data[i]

            predictions = fmodel.predictions(image)
            # This has to be an adversarial image.
            if np.argmax(predictions) != label:
                counter += 1

                # round the adversarial image
                image = rounder.round(image)

                predictions = fmodel.predictions(image)
                # print(np.argmax(predictions), label)
                if np.argmax(predictions) == label:
                    correct += 1
        timing = time.time() - start_time
        with open(args.out_file_name, "a") as out:
            msg = ",".join((str(x) for x in
                            [spacing, correct, counter, correct / counter,
                             timing, rounder.get_average_diff_per_pixel()]))
            print(msg)
            out.write(msg + "\n")
    f.close()


if __name__ == "__main__":
    np.random.seed(31)
    # arguments
    args = get_args()
    args.dataset = "imagenet"  # "cifar10" or "imagenet"
    # args.sample_count_limit = 0

    if torch.cuda.is_available() and args.use_cuda:
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda id not available")
        args.device = torch.device("cpu")

    print("dataset: ", args.dataset)
    header = "spacing, correct, counter, correct rate (%), time (sec), " \
             "avg diff per pixel"
    print(header)
    args.out_file_name = "results_round_defense_after_foolbox_attack_" + args.dataset + ".csv"
    with open(args.out_file_name, "a") as out:
        out.write(header + "\n")

    run(args)
