import time
import torch
import numpy as np


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    # save fft representations of the original and adversarial images to files
    args.save_out = False
    # args.diff_type = "source"  # "source" or "fft"
    args.diff_type = "fft"
    # args.dataset = "cifar10"  # "cifar10" or "imagenet"
    args.dataset = "imagenet"
    # args.dataset = "mnist"
    # args.index = 13  # index of the image (out of 20) to be used
    args.compress_rate = 0
    args.compress_fft_layer = 60
    args.is_fft_compression = True
    args.interpolate = "exp"
    args.use_foolbox_data = False

    if torch.cuda.is_available() and args.use_cuda:
        print("cuda is available")
        args.device = torch.device("cuda")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("cuda id not available")
        args.device = torch.device("cpu")
    # for values_per_channel in [2**x for x in range(1,8,1)]:
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in range(2, 256, 1):
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in [2]:
    #     args.index = 1
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in [8]:
    #     for index in range(0, 17):  # 12, 13, 16
    #         args.index = index
    #         args.values_per_channel = values_per_channel
    #         run(args)
    # for values_per_channel in [8]:
    #     args.index = 13
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in [8]:
    #     args.index = 16
    #     args.values_per_channel = values_per_channel
    #     run(args)
    # for values_per_channel in [8]:
    #     args.values_per_channel = values_per_channel
    #     for index in range(20):
    #         args.index = index
    #         start = time.time()
    #         run(args)
    #         print("elapsed time: ", time.time() - start)
    # for index in range(20):
    #     args.index = index
    #     for values_per_channel in [2**x for x in range(1,8,1)]:
    #         args.values_per_channel = values_per_channel
    #         run(args)
    for interpolate in ["exp", "log", "const", "linear"]:
        args.interpolate = interpolate
        result_file(args)
        for values_per_channel in [8]:
            args.values_per_channel = values_per_channel
            # indexes = index_ranges([(0, 49999)])  # all validation ImageNet
            # print("indexes: ", indexes)
            for index in range(50000):
                args.index = index
                start = time.time()
                run(args)
                print("single run elapsed time: ", time.time() - start)

    print("total elapsed time: ", time.time() - start_time)