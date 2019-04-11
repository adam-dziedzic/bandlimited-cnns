import torch
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.robustness.show_adversarial_fft_pytorch import get_fmodel
import foolbox
import numpy as np
import time
from cnns.nnlib.datasets.transformations.rounding import RoundingTransformation


def run(args):
    start_time = time.time()
    fmodel = get_fmodel(args=args)
    images, labels = foolbox.utils.samples(dataset=args.dataset, index=0,
                                           batchsize=20,
                                           shape=(args.init_y, args.init_x),
                                           data_format='channels_first')
    images = images / 255
    correct = 0
    counter = 0
    sum_difference = 0
    rounder = RoundingTransformation(args.values_per_channel, np.round)
    for image, label in zip(images, labels):
        counter += 1
        round_image = rounder(image)
        sum_difference += np.sum(np.abs(round_image - image))
        predictions = fmodel.predictions(round_image)
        # print(np.argmax(predictions), label)
        if np.argmax(predictions) == label:
            correct += 1
    timing = time.time() - start_time
    with open("results_round_attack.csv", "a") as out:
        msg = ",".join((str(x) for x in
                        [values_per_channel, correct, counter,
                         correct / counter,
                         timing, sum_difference]))
        print(msg)
        out.write(msg + "\n")


if __name__ == "__main__":
    np.random.seed(31)
    # arguments
    args = get_args()
    args.dataset = "cifar10"  # "cifar10" or "imagenet"

    if torch.cuda.is_available() and args.use_cuda:
        print("cuda is available")
        args.device = torch.device("cuda")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("cuda id not available")
        args.device = torch.device("cpu")

    print("dataset: ", args.dataset)
    header = "spacing, correct, counter, correct rate (%), time (sec), " \
             "sum_difference"
    print(header)
    with open("results_round_attack.csv", "a") as out:
        out.write(header + "\n")

    for values_per_channel in [2 ** x for x in range(1, 8)]:
        args.values_per_channel = values_per_channel
        run(args)
