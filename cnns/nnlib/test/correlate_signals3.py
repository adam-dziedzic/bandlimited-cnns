from nnlib.layers import *
from nnlib.load_time_series import load_data
from nnlib.utils.general_utils import *

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
W = len(x)
print("input size: ", W)

y = train_set_x[1]
y = np.array(y, dtype=np.float64)
WW = 100  # filter size
print("filter size: ", WW)
y = y[:WW]

repetitions = 1
exec_number = 1

b = np.array([0])

stride = 1

timings = []
errors = []


def reconstruct_signal(xfft, fft_size):
    xfft = np.concatenate((xfft, np.zeros(max(fft_size // 2 - len(xfft) + 1, 0))))
    xfft = np.concatenate((xfft, np.conj(np.flip(xfft[1:-1], axis=0))))
    return xfft


log_file = "index_back_filter_signal2" + get_log_time() + ".log"
def correlate_signals(x, y, out_len, energy_rate=None, index_back=None):
    """

    :param x: input signal
    :param y: filter
    :param out_len: required output len
    :param energy_rate: compressed to this energy rate
    :param index_back: how many coefficients to remove
    :return: output signal after correlation of signals x and y
    """
    x_len = len(x)
    fft_size = next_power2(x_len)
    xfft = fft(x, fft_size)
    plot_signal(np.abs(xfft), "xfft before compression")
    yfft = fft(y, fft_size)
    if energy_rate is not None or index_back is not None:
        index = preserve_energy_index(xfft, energy_rate, index_back)
        with open(log_file, "a+") as f:
            f.write("index: " + str(index_back) + ";preserved energy input: " + str(
                get_full_energy(xfft[:index]) / get_full_energy(xfft[:fft_size // 2 + 1])) +
                    ";preserved energy filter: " + str(
                get_full_energy(yfft[:index]) / get_full_energy(yfft[:fft_size // 2 + 1])) + "\n")
        xfft = xfft[:index]
        yfft = yfft[:index]
        xfft = reconstruct_signal(xfft, fft_size)
        yfft = reconstruct_signal(yfft, fft_size)
    out = ifft(xfft * np.conj(yfft))

    # plot_signal(out, "out after ifft")
    out = out[:out_len]
    # plot_signal(out, "after truncating to xlen: " + str(x_len))
    return_value = np.real(out)
    return return_value


def compute_energy(xfft):
    squared_abs = np.abs(xfft) ** 2
    full_energy = np.sum(squared_abs)
    return full_energy


def preserve_energy(xfft, energy_rate=None, index_back=None):
    if energy_rate is not None or index_back is not None:
        initial_length = len(xfft)
        half_fftsize = initial_length // 2
        xfft = xfft[0:half_fftsize + 1]
        # print("input xfft len: ", len(xfft))
        if energy_rate is not None:
            squared_abs = np.abs(xfft) ** 2
            full_energy = np.sum(squared_abs)
            current_energy = 0.0
            preserved_energy = full_energy * energy_rate
            index = 0
            while current_energy < preserved_energy and index < len(squared_abs):
                current_energy += squared_abs[index]
                index += 1
        elif index_back is not None:
            index = len(xfft) - index_back
        # print("index back: ", len(xfft) - index)
    return index


def convolve1D_fft_full(x, w, fftsize, out_size, preserve_energy_rate=1.0):
    """
    Convolve inputs x and w using fft.
    :param x: the first signal in time-domain (1 dimensional)
    :param w: the second signal in time-domain (1 dimensional)
    :param fftsize: the size of the transformed signals (in the frequency domain)
    :param out_size: the expected size of the output
    :param preserve_energy_rate: how much energy of the signal to preserve in the frequency domain
    :return: the output of the convolved signal x and w
    """
    xfft = fft(x, fftsize)
    filterfft = fft(w, fftsize)
    if preserve_energy_rate is not None:
        half_fftsize = fftsize // 2
        xfft = xfft[0:half_fftsize + 1]
        filterfft = filterfft[0:half_fftsize + 1]
        squared_abs = np.abs(xfft) ** 2
        full_energy = np.sum(squared_abs)
        current_energy = 0.0
        preserve_energy = full_energy * preserve_energy_rate
        index = 0
        while current_energy < preserve_energy and index < len(squared_abs):
            current_energy += squared_abs[index]
            index += 1
        # print("index: ", index)
        xfft = xfft[:index]
        # plot_signal(np.abs(filterfft), "Before compression in frequency domain")
        full_energy_filter = np.sum(np.abs(filterfft) ** 2)
        filterfft = filterfft[:index]
        # filter_lost_energy_rate = (full_energy_filter - np.sum(np.abs(filterfft) ** 2)) / full_energy_filter
        # print("filter_lost_energy_rate: ", filter_lost_energy_rate)
        # plot_signal(np.abs(filterfft), "After compression in frequency domain")
    # xfft = xfft / norm(xfft)
    # filterfft = filterfft / norm(filterfft)
    out = xfft * np.conj(filterfft)
    out = np.pad(out, (0, fftsize - len(out)), 'constant')
    out = ifft(out)
    out = np.real(out)
    if preserve_energy_rate is not None:
        # out *= len(out)
        out *= 2
    if len(out) < out_size:
        out = np.pad(out, (0, out_size - len(out)), 'constant')
    out = out[:out_size]
    # import matplotlib.pyplot as plt
    # plt.plot(range(0, len(out)), out)
    # plt.title("cross-correlation output fft")
    # plt.xlabel('time')
    # plt.ylabel('Amplitude')
    # plt.show()
    return out


for energy_rate in [None]:
    print("energy rate: ", energy_rate)
    # pad = WW // 2 - 1
    pad = 0
    # padded_x = (np.pad(x, ((0, pad)), 'constant'))
    padded_x = 0
    conv_params = {"pad": pad, "stride": 1}
    naive_out, _ = conv_forward_naive_1D(x.reshape(1, 1, -1), y.reshape(1, 1, -1), [0], conv_params)
    plot_signal(naive_out[0, 0], "naive out")

    out_W = W + 2 * pad - WW + 1
    returned_signal = correlate_signals(x, y, out_W, energy_rate=None, index_back=None)
    fft_size = next_power2(W)
    # returned_signal = convolve1D_fft_full(x, y, fft_size, out_W)

    numpy_result = np.correlate(x, y, mode="valid")
    plot_signal(numpy_result, "numpy_result")
    print("len numpy_result: ", len(numpy_result))
    # conv_forward_naive(x, y, 0, {})

    plot_signals(numpy_result, returned_signal, label_x="numpy", label_y="returned signal")

    # print("zscore rel error input output signal float32: ", rel_error(np.array(zscore(x), dtype=np.float32),
    #                                                            np.array(zscore(returned_signal), dtype=np.float32)))
    # print("zscore rel error input output signal float64: ", rel_error(np.array(zscore(x), dtype=np.float64),
    #                                                            np.array(zscore(returned_signal), dtype=np.float64)))
    #
    # print("zscore abs error input output signal float64: ", abs_error(np.array(zscore(x), dtype=np.float64),
    #                                                            np.array(zscore(returned_signal), dtype=np.float64)))

    print("rel error input output signal float32: ", rel_error(np.array(numpy_result, dtype=np.float32),
                                                               np.array(returned_signal, dtype=np.float32)))
    print("rel error input output signal float64: ", rel_error(np.array(numpy_result, dtype=np.float64),
                                                               np.array(returned_signal, dtype=np.float64)))

    print("abs error input output signal float64: ", abs_error(np.array(numpy_result, dtype=np.float64),
                                                               np.array(returned_signal, dtype=np.float64)))

    for index_back in range(0, fft_size // 2 + 2):
        returned_signal = correlate_signals(x, y, out_W, energy_rate=None, index_back=index_back)
        print("index back,", index_back, ", preserved energy of cross-correlation,", compute_energy(returned_signal) / compute_energy(numpy_result))
        # print("index_back, ", index_back,
        #       ",abs error between expected signal and fft cross-correlated signal (for type float64), ",
        #       abs_error(np.array(numpy_result, dtype=np.float64),
        #                 np.array(returned_signal,
        #                          dtype=np.float64)))
