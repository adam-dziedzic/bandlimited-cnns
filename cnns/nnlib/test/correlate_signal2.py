import numpy as np
from numpy.fft import fft, ifft
from scipy.stats.mstats import zscore

from nnlib.load_time_series import load_data
from nnlib.utils.general_utils import abs_error, rel_error, plot_signals
from nnlib.layers_old import plot_signal
from nnlib.layers import next_power2

np.random.seed(237)
print("correlate signal")

dataset = "50words"
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]

x = train_set_x[0]
x = np.array(x, dtype=np.float64)


def correlate_signal(x, energy_rate=None):
    # plot_signal(x, "input value")
    x_len = len(x)
    print("input length: ", x_len)
    # fft_size = 1 << (2 * x_len - 1).bit_length()
    fft_size = next_power2(x_len)
    xfft = fft(x, fft_size)
    if energy_rate is not None:
        xfft = preserve_energy(xfft, energy_rate)
    cc = ifft(xfft)
    return_value = np.real(cc)
    return_value = return_value[:x_len]
    return return_value


def get_energy_signal(x):
    squared_abs = np.abs(x) ** 2
    return np.sum(squared_abs)


def preserve_energy(xfft, energy_rate=None):
    if energy_rate is not None:
        input = xfft
        plot_signal(np.abs(xfft), "before preserving energy")
        initial_energy = get_energy_signal(xfft)
        initial_length = len(xfft)
        half_fftsize = initial_length // 2 + 1
        xfft = xfft[0:half_fftsize]
        squared_abs = np.abs(xfft) ** 2  # we need squared_abs to calculate the index
        full_energy = np.sum(squared_abs)
        current_energy = 0.0
        preserve_energy = full_energy * energy_rate
        index = 0
        while current_energy < preserve_energy and index < len(squared_abs):
            current_energy += squared_abs[index]
            index += 1
        print("index: ", index)
        # xfft *= (initial_energy / current_energy)
        # xfft = np.concatenate((xfft[:index], np.zeros(initial_length - index)))
        xfft = np.concatenate((xfft[:index], np.zeros(half_fftsize - index)))
        xfft = np.concatenate((xfft, np.flip(np.conj(xfft[1:]), axis=0)))
        plot_signal(np.abs(xfft), "after preserving energy")
        plot_signals(input, xfft, "before (blue) and after (red) preserve energy")
    return xfft

for energy_rate in [1.0]: # None 1.0
    print("energy rate: ", energy_rate)
    # plot_signal(x, "input signal")
    returned_signal = correlate_signal(x, energy_rate=energy_rate)
    # plot_signal(returned_signal, "output signal")
    print("length input: ", len(x))
    print("length output: ", len(returned_signal))
    plot_signals(x, returned_signal)

    print("zscore rel error input output signal float32: ", rel_error(np.array(zscore(x), dtype=np.float32),
                                                               np.array(zscore(returned_signal), dtype=np.float32)))
    print("zscore rel error input output signal float64: ", rel_error(np.array(zscore(x), dtype=np.float64),
                                                               np.array(zscore(returned_signal), dtype=np.float64)))

    print("zscore abs error input output signal float64: ", abs_error(np.array(zscore(x), dtype=np.float64),
                                                               np.array(zscore(returned_signal), dtype=np.float64)))

    print("rel error input output signal float32: ", rel_error(np.array(x, dtype=np.float32),
                                                               np.array(returned_signal, dtype=np.float32)))
    print("rel error input output signal float64: ", rel_error(np.array(x, dtype=np.float64),
                                                               np.array(returned_signal, dtype=np.float64)))

    print("abs error input output signal float64: ", abs_error(np.array(x, dtype=np.float64),
                                                               np.array(returned_signal, dtype=np.float64)))
