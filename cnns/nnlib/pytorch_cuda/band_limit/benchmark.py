import torch
import time
from cnns.nnlib.pytorch_cuda.band_limit.band import Conv2dfftCpp
from cnns.nnlib.pytorch_layers.conv2D_fft import Conv2dfft


def time_it(layer, name=""):
    formating = 1e6 / 1e5
    repetitions = 1

    forward = 0
    backward = 0
    for _ in range(repetitions):
        start = time.time()
        out = layer.forward(x)
        forward += time.time() - start

        start = time.time()
        out.backward(b)
        backward += time.time() - start

    print(name + ': Forward: {:.3f} us | Backward {:.3f} us'.format(
        forward * formating, backward * formating))

def run():
    N, C, H, W = 16, 3, 32, 32
    F = 32
    HH, WW = 3, 3

    if torch.cuda.is_available():
        print("Cuda is available.")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    x = torch.randn(N, C, H, W, device=device)
    y = torch.randn(F, C, HH, WW, device=device)
    b = torch.randn(N, F, H, W, device=device)

    layer_cpp = Conv2dfftCpp(weight_value=y, padding=2)
    layer_python = Conv2dfft(weight_value=y, padding=2)

    time_it(layer=layer_cpp, name="cpp")
    time_it(layer=layer_python, name="python")


if __name__ == "__main__":
    run()