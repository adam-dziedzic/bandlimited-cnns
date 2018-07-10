"""
Custom FFT based convolution that relies on the autograd (a tape-based automatic differentiation library that supports
all differentiable Tensor operations in torch).
"""

import logging
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
consoleLog = logging.StreamHandler()
logger.addHandler(consoleLog)

current_file_name = __file__.split("/")[-1].split(".")[0]
signal_ndim = 1


def next_power2(x):
    """
    :param x: an integer number
    :return: the power of 2 which is the larger than x but the smallest possible
    """
    return torch.pow(2, torch.ceil(torch.log2(torch.tensor(x))))


def preserve_energy_index(xfft, energy_rate=None, index_back=None):
    """
    To which index should we preserve the xfft signal (and discard the remaining coefficients). This is based on the
    provided energy_rate, or if the energy_rate is not provided, then index_back is applied. If none of the params are
    provided, we

    :param xfft: the input signal to be truncated
    :param energy_rate: how much energy should we preserve in the xfft signal
    :param index_back: how many coefficients of xfft should we discard counting from the back of the xfft
    :return: calculated index
    """
    if energy_rate is not None:
        initial_length = xfft.size(0)
        # the signal in frequency domain is symmetric so we can discard one half
        half_fftsize = torch.div(initial_length, 2)
        xfft = xfft[0:half_fftsize + 1]  # we preserve the middle element
        squared = torch.power(xfft, 2)
        full_energy = torch.sum(squared)  # sum of squared values of the signal
        current_energy = 0.0
        preserved_energy = full_energy * energy_rate
        index = 0
        while current_energy < preserved_energy and index < len(squared):
            current_energy += squared[index]
            index += 1
        return index
    elif index_back is not None:
        return len(xfft) - index_back
    return len(xfft)  # no truncation applied to xfft


def correlate_signals(x, y, fft_size, out_size, preserve_energy_rate=None, index_back=None):
    """
    Cross-correlation of the signals: x and y.

    :param x: input signal
    :param y: filter
    :param out_len: required output len
    :param preserve_energy_rate: compressed to this energy rate
    :param index_back: how many coefficients to remove
    :return: output signal after correlation of signals x and y
    """
    xfft = torch.rfft(x, signal_ndim)
    yfft = torch.rfft(y, signal_ndim)
    if preserve_energy_rate is not None or index_back is not None:
        index = preserve_energy_index(xfft, preserve_energy_rate, index_back)
        # with open(log_file, "a+") as f:
        #     f.write("index: " + str(index_back) + ";preserved energy input: " + str(
        #         compute_energy(xfft[:index]) / compute_energy(xfft[:fft_size // 2 + 1])) +
        #             ";preserved energy filter: " + str(
        #         compute_energy(yfft[:index]) / compute_energy(yfft[:fft_size // 2 + 1])) + "\n")
        xfft = xfft[:index]
        yfft = yfft[:index]
        # print("the signal size after compression: ", index)
        xfft = reconstruct_signal(xfft, fft_size)
        yfft = reconstruct_signal(yfft, fft_size)
    out = ifft(xfft * np.conj(yfft))

    # plot_signal(out, "out after ifft")
    out = out[:out_size]
    # plot_signal(out, "after truncating to xlen: " + str(x_len))
    return_value = np.real(out)
    return return_value


class PyTorchConv1d(Module):
    def __init__(self, filter_width, filter=None, bias=None, padding=None, preserve_energy_rate=None):
        """
        1D convolution using FFT implemented fully in PyTorch.

        :param filter_width: the width of the filter
        :param filter_height: the height of the filter
        :param filter: you can provide the initial filter, i.e. filter weights of shape (F, C, WW), where
        F - number of filters, C - number of channels, WW - size of the filter
        :param bias: you can provide the initial value of the bias, of shape (F,)
        :param padding: the padding added to the front and back of the input signal
        :param preserve_energy_rate: the energy of the input to the convolution (both, the input map and filter) that
        have to be preserved (the final length is the length of the longer signal that preserves the set energy rate).

        Regarding, the stride parameter: the number of pixels between adjacent receptive fields in the horizontal and
        vertical directions, it is 1 for the FFT based convolution.
        """
        super(PyTorchConv1d, self).__init__()
        if filter is None:
            self.filter = Parameter(torch.randn(1, 1, filter_width))
        else:
            self.filter = filter
        if bias is None:
            self.bias = Parameter(torch.randn(1))
        else:
            self.bias = bias
        self.padding = padding
        self.preserve_energy_rate = preserve_energy_rate

    def forward(self, input):
        """
        Forward pass of 1D convolution.

        The input consists of N data points with each data point representing a signal (e.g., time-series) of length W.

        We also have the notion of channels in the 1-D convolution. We want to use more than a single filter even for
        the input signal, so the output is a batch with the same size but the number of output channels is equal to the
        number of input filters.

        :param x: Input data of shape (N, C, W), N - number of data points in the batch, C - number of channels, W - the
        width of the signal or time-series (number of data points in a univariate series)
        :param w:
        :param b: biases
        :param conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.
        :return: output data, of shape (N, F, W') where W' is given by: W' = 1 + (W + 2*pad - WW)

         :see:  source short: https://goo.gl/GwyhXz
         full: https://stackoverflow.com/questions/40703751/
         using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

        """
        N, C, W = input.size()
        F, C, WW = self.filter.size()
        fftsize = int(next_power2(float(W + WW - 1)))
        # pad only the dimensions for the time-series (and neither data points nor the channels)
        padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad)), 'constant'))
        out_W = W + 2 * pad - WW + 1
        out = np.zeros([N, F, out_W])
        for nn in range(N):  # For each time-series in the input batch.
            for ff in range(F):  # For each filter in w
                for cc in range(C):
                    out[nn, ff] += correlate_signals(padded_x[nn, cc], w[ff, cc], fftsize,
                                                     out_size=out_W, preserve_energy_rate=preserve_energy_rate,
                                                     index_back=index_back)
                out[nn, ff] += b[ff]  # add the bias term

        cache = (x, w, b, conv_param)
        return out, cache
        # detach so we can cast to NumPy
        result = conv1d(input, filter)

        result += bias
        return output


if __name__ == "__main__":
    torch.manual_seed(231)
    module = PyTorchConv1d(3)
    print("filter and bias parameters: ", list(module.parameters()))
    input = torch.randn(1, 1, 10, requires_grad=True)
    output = module(input)
    print("forward output: ", output)
    output.backward(torch.randn(1, 1, 8))
    print("gradient for the input: ", input.grad)

"""
expected gradient for the input:  tensor([[[[-0.2478, -0.7275, -1.0670,  1.3629,  1.7458, -0.5786, -0.2722,
          0.5767,  0.7379, -0.6335],
        [-1.7113,  0.1839,  1.0434, -3.5176, -1.7056,  1.0892,  2.0054,
          2.3190, -1.6143, -1.3427],
        [-2.4303, -0.1218,  1.9863, -1.6753, -0.3529, -2.4454,  0.4331,
          1.8996,  1.5348, -0.3813],
        [-1.7727, -1.4130,  2.8780, -0.1220, -1.1942,  0.9997, -2.8926,
         -1.4083,  1.1635,  0.9641],
        [ 0.2487,  0.0023,  0.3793, -0.4038,  1.3017,  0.1421, -0.9947,
          0.5084,  0.1511, -2.1860],
        [-0.1263,  1.7602,  3.3994,  0.7883,  0.6831, -0.7291, -0.3211,
          1.8856,  0.3729, -1.2780],
        [-2.1050,  1.8296,  2.4018,  0.5756,  1.3364, -2.9692, -0.4314,
          3.3727,  3.1612, -1.0387],
        [-0.5624, -1.0603,  0.8454,  0.2767,  0.3005,  0.3977, -1.1085,
         -2.7611, -0.4906, -0.1018],
        [ 0.4603, -0.7684,  1.0566, -0.8825,  0.8468,  1.0482,  1.2088,
          0.2836,  0.0993, -0.0322],
        [ 0.0131,  0.4351, -0.3529,  0.2088, -0.3471,  0.3255,  1.6812,
          0.1925, -0.6875,  0.1037]]]])
"""
