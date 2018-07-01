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

xfft = np.fft.fft(x)
filterfft = np.conj(np.fft.fft(corr_filter, len(xfft)))
# element-wise multiplication in the frequency domain
out = xfft * filterfft
# take the inverse of the output from the frequency domain and return the modules of the complex numbers
out = np.fft.ifft(out)
output = np.array(out, np.double)
#output = np.absolute(out)

output = output[:len(standard_corr)]
# print("output of cross-correlation via fft: ", output)
print("is the fft cross_correlation correct: ", np.allclose(output, standard_corr, atol=1e-12))
print("absolute error: ", np.sum(np.abs(output - standard_corr)))
