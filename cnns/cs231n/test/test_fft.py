import os
from numpy.fft import fft, ifft
from scipy.stats.mstats import zscore
import numpy as np
from cs231n.load_time_series import load_data
from cs231n.utils.general_utils import *

np.random.seed(237)
print("test xfft")

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

xfft = fft(x)
plot_signal(xfft, "xfft")
sum_xfft = sum(xfft[1:len(xfft)//2] + xfft[len(xfft)//2 + 1:])
print("sum xfft: ", sum_xfft)
print("the middle element of xfft: ", xfft[len(xfft)//2])


