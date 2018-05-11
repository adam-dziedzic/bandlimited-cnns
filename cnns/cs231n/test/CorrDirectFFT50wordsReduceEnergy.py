# load the data for time-series
import numpy as np
from scipy import signal

from load_time_series import load_data

np.random.seed(231)

dirname = "50words"
datasets = load_data(dirname)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

x = train_set_x[0]
# print("train_set_x[0]: ", x)
print("len of x: ", len(x))

filter_size = 10
corr_filter = np.random.randn(filter_size)

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
    print("index after energy truncation: ", index)
    xfft = xfft[:index]
    filterfft = np.conj(np.fft.fft(corr_filter, len(xfft)))
    # element-wise multiplication in the frequency domain
    out = xfft * filterfft
    # take the inverse of the output from the frequency domain and return the modules of the complex numbers
    out = np.fft.ifft(out)
    output = np.array(out, np.double)
    # output = np.absolute(out)
    print("total output len: ", len(output))
    output = output[:output_len]
    return output

# generate the rates of preserved energy for an input
rates = np.array([x/1000 for x in range(1000, 900, -1)])
errors = []

for index, rate in enumerate(rates):
    # print("output of cross-correlation via fft: ", output)
    print("rate: ", rate)
    output = fft_cross_correlation(x, len(standard_corr), preserve_energy_rate=rate)
    print("is the fft cross_correlation correct: ", np.allclose(output, standard_corr, atol=1e-12))
    error = np.sum(np.abs(output - standard_corr))
    print("absolute error: ", error)
    errors.append(error)
    if error > 1.0e-3:
        break

print("rates: ", rates)
print("errors: ", errors)

import matplotlib.pyplot as plt
my_xticks = rates[:index+1]
# plt.xticks(my_xticks, my_xticks)
plt.plot(my_xticks, errors)
plt.xlabel('Rate of preserved energy')
plt.ylabel('Absolute error')
plt.show()
