# load the data for time-series
import numpy as np
from scipy import signal

from load_time_series import load_data

np.random.seed(231)

x = np.array([1, 2, 3, 4])
# print("train_set_x[0]: ", x)
print("len of x: ", len(x))

filter_size = 2
corr_filter = np.array([1, 2])

standard_corr = signal.correlate(x, corr_filter, 'valid')
print("len of standard corr: ", len(standard_corr))


# print("standard_corr:", standard_corr)

def fft_cross_correlation(x, output_len, preserve_energy_rate=0.95):
    xfft = np.fft.fft(x)
    squared_abs = np.abs(xfft) ** 2
    full_energy = np.sum(squared_abs)
    current_energy = 0.0
    preserve_energy = full_energy * preserve_energy_rate
    index = 0
    while current_energy < preserve_energy and index < len(squared_abs):
        current_energy += squared_abs[index]
        index += 1
    xfft = xfft[:index]
    filterfft = np.conj(np.fft.fft(corr_filter, len(xfft)))
    # element-wise multiplication in the frequency domain
    out = xfft * filterfft
    # take the inverse of the output from the frequency domain and return the modules of the complex numbers
    out = np.fft.ifft(out)
    output = np.array(out, np.double)
    # output = np.absolute(out)

    output = output[:output_len]
    return output


# print("output of cross-correlation via fft: ", output)
output = fft_cross_correlation(x, len(standard_corr), preserve_energy_rate=0.95)
print("is the fft cross_correlation correct: ", np.allclose(output, standard_corr, atol=1e-12))
print("absolute error: ", np.sum(np.abs(output - standard_corr)))
