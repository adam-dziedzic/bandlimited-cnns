# load the data for time-series
import numpy as np
from scipy import signal
import os

from load_time_series import load_data

np.random.seed(231)


def fft_cross_correlation(x, corr_filter, output_len, preserve_energy_rates=0.95):
    xfft = np.fft.fft(x)
    xfft = xfft[1:len(x) // 2]
    # print("length of the input signal: ", len(x))
    # print("modules of xfft: ", np.abs(xfft))
    # import matplotlib.pyplot as plt
    # plt.plot(range(0, len(xfft)), np.abs(xfft))
    # plt.xlabel('index')
    # plt.ylabel('Absolute value')
    # plt.show()

    squared_abs = np.abs(xfft) ** 2
    full_energy = np.sum(squared_abs)
    compressions = []
    for preserve_energy_rate in preserve_energy_rates:
        current_energy = 0.0
        preserve_energy = full_energy * preserve_energy_rate
        index = 0
        while current_energy < preserve_energy and index < len(squared_abs):
            current_energy += squared_abs[index]
            index += 1
        # print("index after energy truncation: ", index)
        xfft = xfft[:index]
        filterfft = np.conj(np.fft.fft(corr_filter, len(xfft)))
        # element-wise multiplication in the frequency domain
        out = xfft * filterfft
        # take the inverse of the output from the frequency domain and return the modules of the complex numbers
        out = np.fft.ifft(out)
        output = np.array(out, np.double)
        # output = np.absolute(out)
        # print("total output len: ", len(output))
        output = output[:output_len]
        compression_ratio = 1 - index / len(x)
        compressions.append(compression_ratio)
        print("energy rate," + str(preserve_energy_rate) + ",compression," + str(compression_ratio) + ",len x," + str(
            len(x)) + ",index," + str(index))
    with open("output_fft_cross_correlation2.csv", mode='a') as file_out:
        file_out.write(dataset + "," + ','.join([str(compression_ratio) for compression_ratio in compressions]) + "\n")
    return output, compressions


# dirname = "50words"
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print("current path: ", dir_path)
all_datasets = os.listdir("../TimeSeriesDatasets")
print("all datasets: ", all_datasets)
print("all datasets len: ", len(all_datasets))
# all_datasets = ["50words"]
# all_datasets = ["CBF"]

for dataset in all_datasets:
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    x = train_set_x[0]
    # print("train_set_x[0]: ", x)
    # print("len of x: ", len(x))

    filter_size = 10
    corr_filter = np.random.randn(filter_size)

    standard_corr = signal.correlate(x, corr_filter, 'valid')
    # print("len of standard corr: ", len(standard_corr))

    # print("standard_corr:", standard_corr)

    # generate the rates of preserved energy for an input
    # rates = np.array([x / 1000 for x in range(1000, 000, -1)])
    rates = np.array([0.995, 0.99, 0.97, 0.95])
    # print("rates: ", rates)
    errors = []
    output, compression = fft_cross_correlation(x, corr_filter, len(standard_corr), preserve_energy_rates=rates)

    # for rate in rates:
    #     # print("output of cross-correlation via fft: ", output)
    #     # print("rate: ", rate)
    #     output, compression = fft_cross_correlation(x, corr_filter, len(standard_corr), preserve_energy_rate=rate)
    #     # print("is the fft cross_correlation correct: ", np.allclose(output, standard_corr, atol=1e-12))
    #     if len(output) < len(standard_corr):
    #         output = np.pad(output, (0, len(standard_corr) - len(output)), "constant")
    #     error = np.sum(np.abs(output - standard_corr))
    #     # print("absolute error: ", error)
    #     errors.append(error)
    #     # if error > 1.0e-3:
    #     #     print(
    #     #         "dataset," + dataset + ",first energy rate for error > 1e-3," + str(rate) + ",compression rate," + str(
    #     #             compression))
    #     #     break
