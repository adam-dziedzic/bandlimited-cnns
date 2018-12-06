import logging
import unittest
import time
import numpy as np
import torch
from torch import tensor
from torch.nn import functional as F
from cnns.nnlib.layers import conv_forward_naive, conv_backward_naive
from cnns.nnlib.pytorch_layers.conv2D_fft \
    import Conv2dfftAutograd, Conv2dfftFunction, Conv2dfft
from cnns.nnlib.pytorch_layers.pytorch_utils import MockContext
from cnns.nnlib.pytorch_layers.pytorch_utils import get_spectrum
from cnns.nnlib.utils.log_utils import get_logger
from cnns.nnlib.utils.log_utils import set_up_logging
from cnns.nnlib.pytorch_layers.test_data.cifar10_image import cifar10_image
from cnns.nnlib.pytorch_layers.test_data.cifar10_lenet_filter import \
    cifar10_lenet_filter
from cnns.nnlib.utils.general_utils import CompressType
from cnns.nnlib.utils.general_utils import NetworkType
from cnns.nnlib.utils.general_utils import StrideType
from cnns.nnlib.utils.general_utils import ConvType
from cnns.nnlib.utils.general_utils import ConvExecType
from cnns.nnlib.utils.general_utils import TensorType
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul4
from cnns.nnlib.pytorch_layers.pytorch_utils import complex_mul5

from cnns.nnlib.utils.arguments import Arguments
from cnns.nnlib.pytorch_architecture.resnet2d import resnet18
from cnns.nnlib.utils.arguments import Arguments

import socket
if socket.gethostname() == "skr-compute1" or socket.gethostname() == "adam-gpu2":
    from complex_mul_cpp import complex_mul as complex_mul_cpp
    # from complex_mul_cuda import complex_mul as complex_mul_cuda
    # from complex_mul_cuda import complex_mul_stride as complex_mul_stride_cuda
    from complex_mul_cuda import \
        complex_mul_stride_no_permute as complex_mul_stride_no_permute_cuda
    from complex_mul_cuda import \
        complex_mul_shared_log as complex_mul_shared_log_cuda

"""
Results:
Testing started at 10:03 PM ...
ssh://ady@skr-compute1:22/home/ady/anaconda3/bin/python3.6 -u /home/ady/.pycharm_helpers/pycharm/_jb_unittest_runner.py --target conv2D_fft_benchmark.TestBenchmarkConv2d.test_mem_usage
Launching unittests with arguments python -m unittest conv2D_fft_benchmark.TestBenchmarkConv2d.test_mem_usage in /local/code/time-series-ml/cnns/nnlib/pytorch_layers
2018-11-27 22:03:53,779 - root - INFO - set_up_logging(19)- started logging to: conv2D_benchmark.log
2018-11-27 22:03:53,779 - conv2D_fft_benchmark - INFO - setUp(31)- Set up test
device used:  cuda
convStandard time:  0.0043790340423583984
fft_forward_time:  0.00822305679321289
backward pass with step
create tensor time:  0.00011348724365234375
correlation time:  0.0046923160552978516
irfft time:  0.012486696243286133
convFFT time:  0.028664827346801758
Pytorch speedup is: 6.545924756356509 X


Ran 1 test in 4.760s

OK
Process finished with exit code 0

Testing started at 11:23 PM ...
ssh://ady@skr-compute1:22/home/ady/anaconda3/bin/python3.6 -u /home/ady/.pycharm_helpers/pycharm/_jb_unittest_runner.py --target conv2D_fft_benchmark.TestBenchmarkConv2d.test_forward_backward
Launching unittests with arguments python -m unittest conv2D_fft_benchmark.TestBenchmarkConv2d.test_forward_backward in /local/code/time-series-ml/cnns/nnlib/pytorch_layers
2018-11-27 23:24:01,603 - root - INFO - set_up_logging(19)- started logging to: conv2D_benchmark.log
2018-11-27 23:24:01,603 - conv2D_fft_benchmark - INFO - setUp(55)- Set up test
device used:  cuda
convStandard time:  0.004107236862182617
fft_forward_time:  0.008202791213989258
backward pass with step
create tensor time:  0.00011038780212402344
correlation time:  0.004713535308837891
restore time:  4.76837158203125e-06
irfft time:  0.012326240539550781
convFFT time:  0.02847886085510254
Pytorch forward pass speedup is: 6.9338248098914494 X
standard back time:  0.010608196258544922
total multiply time:  0.00040841102600097656
total restore time:  5.9604644775390625e-06
total multiply time:  0.0008776187896728516
total restore time:  1.3589859008789062e-05
conv fft back time:  0.4154634475708008
Pytorch speedup for backprop: 39.164381714388455 X

**Run the convolution forward pass 10000X** 
Testing started at 6:14 PM ...
ssh://ady@skr-compute1:22/home/ady/anaconda3/bin/python3.6 -u /home/ady/.pycharm_helpers/pycharm/_jb_unittest_runner.py --target conv2D_fft_benchmark.TestBenchmarkConv2d.test_forward_timing
Launching unittests with arguments python -m unittest conv2D_fft_benchmark.TestBenchmarkConv2d.test_forward_timing in /local/code/time-series-ml/cnns/nnlib/pytorch_layers
2018-11-29 18:15:11,832 - root - INFO - set_up_logging(19)- started logging to: conv2D_benchmark.log
2018-11-29 18:15:11,832 - conv2D_fft_benchmark - INFO - setUp(82)- Set up test
device used:  cuda
convStandard time:  6.664342880249023
convFFT time:  335.75190138816833
Pytorch speedup is: 50.38034618285163 X


Ran 1 test in 346.483s

OK
Process finished with exit code 0

device used:  cuda
convStandard time:  0.07773208618164062
global fft time:  0.8000748157501221
preserve energy time total:  1.1724259853363037
global irfft time:  0.6015880107879639
global correlation time:  2.4582550525665283
convFFT time:  4.650243282318115
Pytorch speedup is: 59.823986602542085 X

preserve energy: 50
global fft time:  0.8730700016021729
preserve energy time total:  1.0860843658447266
global complex time:  0.930959939956665
global irfft time:  0.5697360038757324
global correlation time:  1.7040534019470215
convFFT time:  3.8487296104431152
Pytorch speedup is: 58.458966169089365 X

device used:  cuda
preserve energy:  100
min_batch_size (equivalent to the batch slice for fft):  128
global fft time:  0.8541977405548096
global complex time:  1.8472542762756348
global irfft time:  0.5717365741729736
global correlation time:  2.4818029403686523
convFFT time:  3.5268049240112305
Pytorch speedup is: 59.6344007127509 X

device used:  cuda
preserve energy:  100
min_batch_size (equivalent to the batch slice for fft):  128
next power 2:  True
convStandard time:  0.07816290855407715
global fft time:  0.1257774829864502
global complex time:  2.027482271194458
global irfft time:  0.15481352806091309
global correlation time:  2.260103225708008
convFFT time:  2.583146095275879
Pytorch speedup is: 33.0482340417095 X

global_block_conv1_time:  0.0033943653106689453
global_block_conv1_time:  1.5696039199829102

energy: 50%
global_block_conv1_time:  0.0033521652221679688
standard eval time:  0.02388596534729004
standard layer1 cumulative time:  0.004895925521850586
global_block_conv1_time:  0.15587091445922852
global fft time:  0.041803598403930664
preserve energy time total:  0.062485456466674805
global complex time:  0.16770434379577637
global irfft time:  0.021817684173583984
global correlation time:  0.21394872665405273
conv2D FFT time:  0.35585856437683105
fft layer1 cumulative time:  0.045282602310180664
pytorch speedup over fft for testing resnet18:  14.89822827768628
pytorch speedup over fft for layer 1:  9.249038227416605

energy: 80%
global_block_conv1_time:  0.0033524036407470703
standard eval time:  0.023734569549560547
standard layer1 cumulative time:  0.004805088043212891
global_block_conv1_time:  0.20783185958862305
global fft time:  0.04039120674133301
preserve energy time total:  0.060901641845703125
global complex time:  0.30475831031799316
global irfft time:  0.018187522888183594
global correlation time:  0.34923863410949707
conv2D FFT time:  0.48766398429870605
fft layer1 cumulative time:  0.0490567684173584
pytorch speedup over fft for testing resnet18:  20.54656956303365
pytorch speedup over fft for layer 1:  10.20933809665575

energy: 90%
global_block_conv1_time:  0.003366231918334961
standard eval time:  0.02382636070251465
standard layer1 cumulative time:  0.004833221435546875
global_block_conv1_time:  0.29797935485839844
global fft time:  0.04062294960021973
preserve energy time total:  0.06063055992126465
global complex time:  0.44714999198913574
global irfft time:  0.022483348846435547
global correlation time:  0.5007925033569336
conv2D FFT time:  0.6423120498657227
fft layer1 cumulative time:  0.07581090927124023
pytorch speedup over fft for testing resnet18:  26.95804272777305
pytorch speedup over fft for layer 1:  15.685378847671666

energy 95%:
global_block_conv1_time:  0.003363370895385742
standard eval time:  0.023844003677368164
standard layer1 cumulative time:  0.004848480224609375
global_block_conv1_time:  0.4710414409637451
global fft time:  0.04078269004821777
preserve energy time total:  0.06490635871887207
global complex time:  0.7352702617645264
global irfft time:  0.01799488067626953
global correlation time:  0.7867190837860107
conv2D FFT time:  0.9344315528869629
fft layer1 cumulative time:  0.1287529468536377
pytorch speedup over fft for testing resnet18:  39.18937295643392
pytorch speedup over fft for layer 1:  26.555320613690007

energy 100%:
standard eval time:  0.027710437774658203
standard layer1 cumulative time:  0.004811525344848633
global fft time:  0.0032012462615966797
global_block_conv1_time:  1.5661354064941406
global fft time:  0.04030442237854004
global complex time:  1.4989120960235596
global irfft time:  0.018094539642333984
global correlation time:  2.169558048248291
conv2D FFT time:  2.256150007247925
fft layer1 cumulative time:  0.33473920822143555
pytorch speedup over fft for testing resnet18:  81.41877893070398
pytorch speedup over fft for layer 1:  69.57028888558546
"""


class TestBenchmarkConv2d(unittest.TestCase):

    def setUp(self):
        log_file = "conv2D_benchmark.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def test_forward_correctness(self):
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device used: ", str(device))
        N, C, H, W = 128, 16, 32, 32
        K, HH, WW = 16, 3, 3
        x = torch.randn(N, C, H, W, dtype=dtype, device=device)
        y = torch.randn(K, C, HH, WW, dtype=dtype, device=device)

        repetitions = 1

        start = time.time()
        for repeat in range(repetitions):
            convStandard = torch.nn.functional.conv2d(input=x, weight=y,
                                                      stride=1)
        convStandardTime = time.time() - start
        print("convStandard time: ", convStandardTime)

        conv = Conv2dfftFunction()
        start = time.time()
        for repeat in range(repetitions):
            convFFT = conv.forward(ctx=None, input=x, filter=y, stride=1,
                                   args=Arguments(
                                       stride_type=StrideType.STANDARD))
        convFFTtime = time.time() - start
        print("convFFT time: ", convFFTtime)
        speedup = convFFTtime / convStandardTime
        print(f"Pytorch speedup is: {speedup} X")

        np.testing.assert_array_almost_equal(
            x=convStandard.cpu().detach().numpy(),
            y=convFFT.cpu().detach().numpy(), decimal=3,
            err_msg="The expected array x and computed y are not almost equal.")

    def test_forward_timing(self):
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device used: ", str(device))
        N, C, H, W = 128, 16, 32, 32
        K, HH, WW = 16, 3, 3
        x = torch.randn(N, C, H, W, dtype=dtype, device=device)
        y = torch.randn(K, C, HH, WW, dtype=dtype, device=device)

        repetitions = 100
        min_batch_size = 128
        preserve_energy = 100
        stride = 1
        next_power2 = True

        print("preserve energy: ", preserve_energy)
        print("min_batch_size (equivalent to the batch slice for fft): ",
              min_batch_size)
        print("next power 2: ", next_power2)

        convStandard = torch.nn.Conv2d(in_channels=C, out_channels=K,
                                       kernel_size=(HH, WW), stride=stride)
        convStandard.to(device)

        start = time.time()
        for repeat in range(repetitions):
            convStandard.forward(x)
        convStandardTime = time.time() - start
        print("convStandard time: ", convStandardTime)

        conv = Conv2dfft(weight_value=y, stride=stride,
                         args=Arguments(stride_type=StrideType.STANDARD,
                                        min_batch_size=min_batch_size,
                                        preserve_energy=preserve_energy,
                                        next_power2=next_power2))
        conv.to(device)
        start = time.time()
        for repeat in range(repetitions):
            conv.forward(input=x)
        convFFTtime = time.time() - start
        print("convFFT time: ", convFFTtime)
        speedup = convFFTtime / convStandardTime
        print(f"Pytorch speedup is: {speedup} X")

    def test_forward_compression(self):
        dtype = torch.float
        if torch.cuda.is_available():
            print("cuda is available")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device used: ", str(device))
        N, C, H, W = 128, 16, 32, 32
        K, HH, WW = 16, 3, 3
        x = torch.randn(N, C, H, W, dtype=dtype, device=device)
        y = torch.randn(K, C, HH, WW, dtype=dtype, device=device)

        start = time.time()
        torch.nn.functional.conv2d(input=x, weight=y, stride=1)
        convStandardTime = time.time() - start
        print("convStandard time: ", convStandardTime)

        conv = Conv2dfftFunction()
        start = time.time()
        conv.forward(ctx=None, input=x, filter=y, stride=1,
                     args=Arguments(stride_type=StrideType.STANDARD,
                                    preserve_energy=80))
        convFFTtime = time.time() - start
        print("convFFT time: ", convFFTtime)
        speedup = convFFTtime / convStandardTime
        print(f"Pytorch speedup is: {speedup} X")

    def test_forward_backward(self):
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device used: ", str(device))
        N, C, H, W = 128, 16, 32, 32
        K, HH, WW = 16, 3, 3
        x = torch.randn(N, C, H, W, dtype=dtype, device=device,
                        requires_grad=True)
        x_expect = x.clone().detach().requires_grad_(True)
        y = torch.randn(K, C, HH, WW, dtype=dtype, device=device,
                        requires_grad=True)
        y_expect = y.clone().detach().requires_grad_(True)
        start = time.time()
        convStandard = torch.nn.functional.conv2d(input=x_expect,
                                                  weight=y_expect, stride=1)
        convStandardTime = time.time() - start
        print("convStandard time: ", convStandardTime)

        conv = Conv2dfft(weight_value=y, stride=1, bias=False,
                         args=Arguments(stride_type=StrideType.STANDARD))
        start = time.time()
        convFFT = conv.forward(input=x)
        convFFTtime = time.time() - start
        print("convFFT time: ", convFFTtime)
        speedup = convFFTtime / convStandardTime
        print(f"Pytorch forward pass speedup is: {speedup} X")

        np.testing.assert_array_almost_equal(
            x=convStandard.cpu().detach().numpy(),
            y=convFFT.cpu().detach().numpy(), decimal=3,
            err_msg="The expected array x and computed y are not almost equal.")

        dout = torch.randn(list(convStandard.size()), device=device,
                           dtype=dtype)
        dout_clone = dout.clone()

        standard_back_time_start = time.time()
        convStandard.backward(dout)
        standard_back_time = time.time() - standard_back_time_start
        print("standard back time: ", standard_back_time)

        fft_back_time_start = time.time()
        convFFT.backward(dout_clone)
        conv_fft_back_time = time.time() - fft_back_time_start
        assert conv.is_manual[0] == 1
        print("conv fft back time: ", conv_fft_back_time)
        speedup = conv_fft_back_time / standard_back_time
        print(f"Pytorch speedup for backprop: {speedup} X")

        np.testing.assert_array_almost_equal(x.grad.cpu().detach().numpy(),
                                             x_expect.grad.cpu().detach().numpy(),
                                             decimal=3)

        np.testing.assert_array_almost_equal(y.grad.cpu().detach().numpy(),
                                             y_expect.grad.cpu().detach().numpy(),
                                             decimal=3)

    def test_forward_pass_resnet18(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("\ndevice used: ", str(device))

        C = 3
        # dtype = torch.float
        # random mini batch imitating cifar-10
        # N, H, W = 128, 32, 32
        # inputs = torch.randn(N, C, H, W, dtype=dtype, device=device,
        #                      requires_grad=True)
        args = Arguments()
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        args.sample_count_limit = 128
        args.min_batch_size = 128
        args.test_batch_size = args.min_batch_size
        args.network_type = NetworkType.ResNet18
        from cnns.nnlib.datasets.cifar10 import get_cifar10
        train_loader, test_loader = get_cifar10(args)

        repetition = 1

        args.in_channels = 3
        # args.conv_type = "FFT2D"
        args.conv_type = ConvType.STANDARD2D
        args.index_back = None
        args.preserve_energy = 100
        args.is_debug = False
        args.next_power2 = True
        args.compress_type = CompressType.STANDARD
        args.tensor_type = TensorType.FLOAT32
        args.num_classes = 10
        args.min_batch_size = 0
        args.test_batch_size = args.min_batch_size
        args.in_channels = C

        model = resnet18(args=args)
        model.to(device)
        model.eval()
        start_eval = time.time()
        for _ in range(repetition):
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                outputs_standard = model(inputs)
        standard_time = time.time() - start_eval
        print("total time with pytorch conv2D: ", standard_time)
        # layer1_standard = model.global_layer1_time
        # print("standard layer1 cumulative time: ", layer1_standard)

        # print("outputs standard: ", outputs_standard)

        args.conv_type = ConvType.FFT2D
        args.conv_exec_type = ConvExecType.CUDA_SHARED_LOG
        # args.conv_exec_type = ConvExecType.CUDA
        model = resnet18(args=args)
        model.to(device)
        model.eval()
        start_eval = time.time()
        for _ in range(repetition):
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                outputs_fft = model(inputs)
        fft_time1 = time.time() - start_eval
        print("total time with cuda shared log based conv2D: ", fft_time1)

        args.conv_type = ConvType.FFT2D
        args.conv_exec_type = ConvExecType.CUDA
        # args.conv_exec_type = ConvExecType.CUDA
        model = resnet18(args=args)
        model.to(device)
        model.eval()
        start_eval = time.time()
        for _ in range(repetition):
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                outputs_fft = model(inputs)
        fft_time2 = time.time() - start_eval
        print("total time with cuda stride no permute based conv2D: ", fft_time2)
        # layer1_fft = model.global_layer1_time
        # print("fft layer1 cumulative time: ", layer1_fft)

        # print("outputs fft: ", outputs_fft)

        print("pytorch speedup over cuda shared log for testing ResNet-18: ",
              fft_time1 / standard_time)
        print("pytorch speedup over cuda stride no permute for testing ResNet-18: ",
              fft_time2 / standard_time)
        # print("pytorch speedup over fft for layer 1: ",
        #       layer1_fft / layer1_standard)

    def test_complex_mul(self):
        N, C, H, W, I = 512, 64, 2, 2, 2
        F = 128  # number of filter banks
        repetitions = 100
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = torch.randn(N, C, H, W, I, dtype=dtype, device=device)
        y = torch.randn(F, C, H, W, I, dtype=dtype, device=device)

        start_mul_time = time.time()
        if torch.cuda.is_available():
            for _ in range(repetitions):
                out = torch.empty(N, F, H, W, I, dtype=dtype, device=device)
                complex_mul_stride_no_permute_cuda(x, y, out, 1024)
        cuda_time = time.time() - start_mul_time
        print("\ncuda multiply time: ", cuda_time)

        x = x.unsqueeze(dim=1)
        start_mul_time = time.time()
        for _ in range(repetitions):
            complex_mul(x, y)
        pytorch_time = time.time() - start_mul_time
        print("pytorch multiply time: ", pytorch_time)

        print("cuda speedup is: ", pytorch_time/cuda_time)

    def test_complex_mul_torch_vs_numpy(self):
        N, C, H, W, I = 128, 3, 32, 32, 2
        K = 16  # number of filter banks
        repetitions = 10
        dtype = torch.float
        device = torch.device("cpu")
        x = torch.randn(N, 1, C, H, W, I, dtype=dtype, device=device)
        y = torch.randn(K, C, H, W, I, dtype=dtype, device=device)
        x_np = x.numpy()
        x_complex = x_np[..., :1] + 1.0j * x_np[..., 1:]
        y_np = y.numpy()
        y_complex = y_np[..., :1] + 1.0j * y_np[..., 1:]
        start_mul_time = time.time()
        # out = torch.empty(N, K, C, H, W, I, dtype=dtype, device=device)
        for _ in range(repetitions):
            result_torch = complex_mul(x, y)
        print("multiplication time: ", time.time() - start_mul_time)

        start_mul_np = time.time()
        for _ in range(repetitions):
            result_np = x_complex * y_complex
        print("numpy multiplication time: ", time.time() - start_mul_np)

        np.testing.assert_array_almost_equal(result_torch, np.concatenate(
            (result_np.real, result_np.imag), axis=-1), decimal=5)

    def test_complex_mul_with_out_tensor(self):
        N, C, H, W, I = 128, 3, 32, 32, 2
        K = 16  # number of filter banks
        repetitions = 100
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = torch.randn(N, 1, C, H, W, I, dtype=dtype, device=device)
        y = torch.randn(K, C, H, W, I, dtype=dtype, device=device)
        start_mul_time = time.time()
        out = torch.empty(N, K, C, H, W, I, dtype=dtype, device=device)
        for _ in range(repetitions):
            complex_mul5(x, y, out)
        print("complex mul out multiplication time: ",
              time.time() - start_mul_time)

    def test_complex_mul_cpp(self):
        N, C, H, W, I = 128, 3, 32, 32, 2
        K = 16  # number of filter banks
        repetitions = 1000
        dtype = torch.float
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = torch.randn(N, 1, C, H, W, I, dtype=dtype, device=device)
        y = torch.randn(K, C, H, W, I, dtype=dtype, device=device)
        start_mul_time = time.time()
        for _ in range(repetitions):
            out = complex_mul_cpp(x, y)
        stop_mul_time = time.time()
        print("complex mul cpp multiplication time: ",
              stop_mul_time - start_mul_time)

        expect = complex_mul(x, y)
        expect = expect.cpu().numpy()
        out = out.cpu().numpy()
        np.testing.assert_array_almost_equal(expect, out, decimal=5)
