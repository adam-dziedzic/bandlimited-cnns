import torch
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.robustness import \
    get_foolbox_model
import numpy as np
import time
from cnns.nnlib.datasets.cifar import get_cifar, cifar_mean, cifar_std
from cnns.nnlib.robustness import unnormalize
from cnns.nnlib.robustness import normalize


def run(args):
    start_time = time.time()
    model_path = "2019-01-14-15-36-20-089354-dataset-cifar10-preserve-energy-100.0-test-accuracy-93.48-compress-rate-0-resnet18.model"
    compress_rate = 0
    fmodel = get_foolbox_model(args, model_path=model_path,
                               compress_rate=compress_rate)
    correct = 0
    counter = 0
    sum_difference = 0
    round_multiplier = 255 // spacing
    ext_muliplier = 1.0 / round_multiplier
    for batch_idx, (data, target) in enumerate(test_loader):
        # print("batch_idx: ", batch_idx)
        for i, label in enumerate(target):
            counter += 1
            label = label.item()
            image = data[i].numpy()

            # round the image
            mean = np.array(cifar_mean, dtype=np.float32).reshape((3, 1, 1))
            std = np.array(cifar_std, dtype=np.float32).reshape((3, 1, 1))
            image = unnormalize(image, mean, std)
            # print("image max min: ", np.max(image), np.min(image))
            round_image = ext_muliplier * np.round(round_multiplier * image)
            sum_difference += np.sum(np.abs(round_image - image))
            image = normalize(round_image, mean, std)

            predictions = fmodel.predictions(image)
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

    args.sample_count_limit = 0
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

    # for spacing in [2 ** x for x in range(0, 8)]:
    for spacing in range(1, 256):
        args.spacing = spacing
        run(args)
