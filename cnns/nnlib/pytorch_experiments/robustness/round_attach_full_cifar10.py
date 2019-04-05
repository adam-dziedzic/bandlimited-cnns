import torch
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.pytorch_experiments.robustness.show_adversarial_fft_pytorch import \
    get_fmodel
import numpy as np
import time
from cnns.nnlib.datasets.cifar import get_cifar


def run(args):
    start_time = time.time()
    fmodel = get_fmodel(args=args)
    correct = 0
    counter = 0
    sum_difference = 0
    round_multiplier = 255 // spacing
    ext_muliplier = 1.0 / round_multiplier
    for batch_idx, (data, target) in enumerate(test_loader):
        # print("batch_idx: ", batch_idx)
        for i, label in enumerate(target):
            label = label.item()
            image = data[i].numpy()
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
    args.dataset = "cifar10"

    args.sample_count_limit = 20
    train_loader, test_loader, train_dataset, test_dataset = get_cifar(args,
                                                                       args.dataset)

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
