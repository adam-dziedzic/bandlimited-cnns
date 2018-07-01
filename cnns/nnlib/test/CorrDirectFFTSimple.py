# load the data for time-series
import numpy as np
from scipy import signal

from nnlib.load_time_series import load_data

np.random.seed(231)

x = np.array([1, 2, 3, 4])
# print("train_set_x[0]: ", x)
print("len of x: ", len(x))

filter_size = 2
# corr_filter = np.random.randn(filter_size)
corr_filter = np.array([1, 2])

standard_corr = signal.correlate(x, corr_filter, 'valid')
print("len of standard corr: ", len(standard_corr))

print("standard_corr:", standard_corr)

xfft = np.fft.fft(x)
filterfft = np.conj(np.fft.fft(corr_filter, len(xfft)))
# element-wise multiplication in the frequency domain
out = xfft * filterfft
# take the inverse of the output from the frequency domain and return the modules of the complex numbers
out = np.fft.ifft(out)
#output = np.array(out, np.double)[:len(standard_corr)]
# output = np.real(out)
output = np.abs(out)
output = output[:len(standard_corr)]
# output = np.absolute(out)

print("output of cross-correlation via fft: ", output)
