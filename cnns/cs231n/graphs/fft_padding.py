import numpy as np
from numpy.fft import fft

from cs231n.load_time_series import load_data
from cs231n.utils.general_utils import abs_error

np.random.seed(231)

# dataset = "Adiac"
dataset = "50words"
# dataset = "Herring"
# dataset = "InlineSkate"
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

x = train_set_x[0]
filter_size = 4
full_filter = train_set_x[1]
filters = full_filter[:filter_size]

xfft1 = fft(x)
xfft2 = fft(x, 2 * len(x))
# print("abs error xfft1 vs xfft2: ", abs_error(xfft1, xfft2))
pad = len(x) // 2
padded_x = (np.pad(x, (pad, pad), 'constant'))
xfft3 = fft(padded_x)
print("total abs error xfft2 vs xfft3: ", abs_error(xfft2, xfft3))
print("total energy of xfft2 and xfft3: ", np.sum(np.abs(xfft2) + np.abs(xfft3)))
print("relative abs error xfft2 vs xfft3: ", abs_error(xfft2, xfft3) / np.sum(np.abs(xfft2) + np.abs(xfft3)))
print("error in magnitudes between xfft2 vs xfft3: ", np.sum(np.abs(np.abs(xfft2)-np.abs(xfft3))))
import matplotlib.pyplot as plt

# plt.plot(range(0, len(xfft1)), np.abs(xfft1), color="red")
plt.plot(range(0, len(xfft2)), np.abs(xfft2), color="red", linestyle="dotted")
plt.plot(range(0, len(xfft3)), np.abs(xfft3), color="green", linestyle="dotted")
plt.title("cross-correlation output cross_correlate")
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.show()
