from cs231n.layers import compress_fft_1D

import os
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

all_datasets = os.listdir("../TimeSeriesDatasets")
# all_datasets = ["synthetic_control", "50words"]
# all_datasets = ["BirdChicken"]
all_datasets = ["50words"]
for dataset in all_datasets:
# for signal_length in range(270, 10, -2):
# for signal_length in range(10, 271, 2):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    x = train_set_x[0]
    # x = x[:signal_length]
    x = np.array(x, dtype=np.float64)
    signal_length = len(x)
    plot_signal(x, "input signal")
    # the length of the output signal
    for y_len in range(100, 101):
        y = compress_fft_1D(x, y_len=y_len)
        plot_signals(x, y, title="before/after signal compression")
        print("ratio of energy y to energy of x: ", energy(y) / energy(x))
