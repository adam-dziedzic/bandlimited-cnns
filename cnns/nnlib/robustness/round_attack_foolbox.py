import torch
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.robustness import \
    get_fmodel
import foolbox
import numpy as np
import time


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
    round_multiplier = 255 // spacing
    ext_muliplier = 1.0 / round_multiplier
    for image, label in zip(images, labels):
        counter += 1
        round_image = ext_muliplier * np.round(round_multiplier * image)
        sum_difference += np.sum(np.abs(round_image - image))
        predictions = fmodel.predictions(round_image)
        # print(np.argmax(predictions), label)
        if np.argmax(predictions) == label:
            correct += 1
    timing = time.time() - start_time
    with open("results_round_attack.csv", "a") as out:
        msg = ",".join((str(x) for x in
                        [spacing, correct, counter, correct / counter,
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

    for spacing in [2 ** x for x in range(0, 8)]:
        args.spacing = spacing
        run(args)
