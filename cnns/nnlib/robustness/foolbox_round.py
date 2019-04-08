from cnns import matplotlib_backend

print("Using:", matplotlib_backend.backend)
import torch
from cnns.nnlib.utils.exec_args import get_args
import numpy as np
import time
from cnns.nnlib.datasets.cifar import get_cifar
import foolbox
from cnns.nnlib.robustness.utils import get_foolbox_model
from cnns.nnlib.robustness.uitls import Rounder


def run(args):
    start_time = time.time()
    model_path = "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"
    compress_rate = 0
    fmodel = get_foolbox_model(args, model_path=model_path,
                               compress_rate=compress_rate)
    attack = foolbox.attacks.FGSM(fmodel)
    correct = 0
    counter = 0
    rounder = Rounder(spacing)
    for batch_idx, (data, target) in enumerate(test_loader):
        # print("batch_idx: ", batch_idx)
        for i, label in enumerate(target):
            # counter += 1
            label = label.item()
            image = data[i].numpy()

            # adversarial attack (successful if returned image is not None)
            image = attack(image, label)

            # if image is None:
            #     correct += 1
            if image is not None:
                counter += 1
                # round the adversarial image
                image = rounder.round(image)

                predictions = fmodel.predictions(image)
                # print(np.argmax(predictions), label)
                if np.argmax(predictions) == label:
                    correct += 1
    timing = time.time() - start_time
    with open("results_round_attack.csv", "a") as out:
        msg = ",".join((str(x) for x in
                        [spacing, correct, counter, correct / counter,
                         timing, rounder.get_average_diff_per_pixel()]))
        print(msg)
        out.write(msg + "\n")


if __name__ == "__main__":
    np.random.seed(31)
    # arguments
    args = get_args()
    args.dataset = "cifar10"

    args.sample_count_limit = 10
    train_loader, test_loader, train_dataset, test_dataset = get_cifar(args,
                                                                       args.dataset)

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
    with open("results_round_attack_after_foolbox.csv", "a") as out:
        out.write(header + "\n")

    for spacing in [2 ** x for x in range(0, 8)]:
        # for spacing in range(1,256):
        args.spacing = spacing
        run(args)
