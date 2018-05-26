import os
from numpy.fft import fft, ifft
from scipy.stats.mstats import zscore

from cs231n.load_time_series import load_data
from cs231n.utils.general_utils import *

np.random.seed(237)
print("correlate signal")

# dataset = "Adiac"
dataset = "50words"
# dataset = "Herring"
# dataset = "InlineSkate"
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]

x = train_set_x[0]
x = np.array(x, dtype=np.float64)

repetitions = 1
exec_number = 1

b = np.array([0])

stride = 1

timings = []
errors = []


def correlate_signal(x, index_back=0):
    # plot_signal(x, "input value")
    x_len = len(x)
    # print("input x size: ", len(x))
    # fft_size = 1 << (2 * x_len - 1).bit_length()
    fft_size = next_power2(x_len)
    # print("fft size: ", fft_size)
    xfft = fft(x, fft_size)
    # print("xfft: ", xfft)
    # plot_signal(xfft, "initial xfft")
    xfft = preserve_energy(xfft, index_back)
    # plot_signal(xfft, "xfft after compression")
    zero_padding = fft_size // 2 - len(xfft) + 1
    if zero_padding > 0:
        xfft = np.concatenate((xfft, np.zeros(zero_padding, dtype=complex)))
    # print("length of fft after padding with zeros: ", len(xfft))
    xfft = np.concatenate((xfft, np.conj(np.flip(xfft[1:-1], axis=0))))
    # print("length of fft after reconstruction: ", len(xfft))
    # plot_signal(xfft, "xfft after reconstruction")
    cc = ifft(xfft)
    # print("cc length: ", len(cc))
    # plot_signal(cc, "first cc after ifft")
    # cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
    cc = cc[:x_len]
    # print("plot cc signal after truncation: ")
    # plot_signal(cc)
    return_value = np.real(cc)
    return_value = return_value[-x_len:]
    # plot_signal(return_value, "returned value")
    return return_value


def preserve_energy(xfft, index_back=0):
    initial_length = len(xfft)
    half_fftsize = initial_length // 2
    # first_half = xfft[1:half_fftsize]
    # second_half = np.conj(np.flip(xfft[half_fftsize + 1:], axis=0))
    # print("are close: ", np.allclose(first_half, second_half))
    # print("rel error: ", rel_error(first_half, second_half))
    xfft = xfft[0:half_fftsize + 1]
    xfft = xfft[:len(xfft) - index_back]
    return xfft

all_datasets = os.listdir("../TimeSeriesDatasets")
#all_datasets = ["synthetic_control", "50words"]
all_datasets = ["BirdChicken"]
for dataset in all_datasets:
# for signal_length in range(270, 10, -2):
# for signal_length in range(10, 271, 2):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    x = train_set_x[0]
    # x = x[:signal_length]
    x = np.array(x, dtype=np.float64)
    signal_length = len(x)
    for index_back in range(1, 2):
        returned_signal = correlate_signal(x, index_back=index_back)
        plot_signals(x, returned_signal)
        print("index back,", index_back, ",signal_length,", signal_length, "dataset", dataset,
            ",rel error input output signal float32,", rel_error(np.array(zscore(x), dtype=np.float32),
                                                                 np.array(zscore(returned_signal), dtype=np.float32)),
            ",rel error input output signal float64,", rel_error(np.array(zscore(x), dtype=np.float64),
                                                                 np.array(zscore(returned_signal), dtype=np.float64)),
            ",abs error input output signal float64,", abs_error(np.array(zscore(x), dtype=np.float64),
                                                                 np.array(zscore(returned_signal), dtype=np.float64)))

