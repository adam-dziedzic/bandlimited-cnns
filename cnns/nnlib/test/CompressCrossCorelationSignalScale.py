import matplotlib.patches as mpatches
from pandas import DataFrame
from scipy.stats.mstats import zscore

from nnlib.layers import *
from nnlib.layers_old import *
from nnlib.load_time_series import load_data
from nnlib.utils.general_utils import reshape_3d_rest, abs_error, rel_error
from nnlib.utils.perf_timing import wrapper, timeitrep
import sys

np.random.seed(237)

# x = [-1.0441e+00 -9.2538e-01 -7.1503e-01 -4.8242e-01 -2.6994e-01 -8.8979e-02
#   6.2874e-02  1.9860e-01  3.2979e-01  4.5426e-01  5.5685e-01  6.3188e-01
#   6.7699e-01  6.9004e-01  6.7324e-01  6.2843e-01  5.6135e-01  4.7251e-01
#   3.6990e-01  2.6389e-01  1.6607e-01  8.4425e-02  1.5710e-02 -4.1070e-02
#  -8.8542e-02 -1.2707e-01 -1.5675e-01 -1.7866e-01 -1.7574e-01 -1.3100e-01
#  -3.0890e-02  1.0512e-01  2.2760e-01  2.8950e-01  2.7594e-01  2.2595e-01
#   1.7856e-01  1.5172e-01  1.3680e-01  1.2414e-01  1.1269e-01  1.0040e-01
#   9.0951e-02  8.2168e-02  7.5933e-02  7.6022e-02  8.1221e-02  8.4768e-02
#   7.9035e-02  7.1215e-02  6.2236e-02  4.6974e-02  2.2526e-02 -1.3913e-02
#  -6.8629e-02 -1.5471e-01 -2.5964e-01 -3.5717e-01 -4.2588e-01 -4.6650e-01
#  -4.9434e-01 -5.1401e-01 -5.2389e-01 -5.1674e-01 -4.9652e-01 -4.7229e-01
#  -4.5138e-01 -4.4132e-01 -4.4166e-01 -4.5155e-01 -4.6451e-01 -4.8018e-01
#  -4.9944e-01 -5.1881e-01 -5.3840e-01 -5.5841e-01 -5.8399e-01 -6.0827e-01
#  -6.3053e-01 -6.5460e-01 -6.8565e-01 -7.2264e-01 -7.5414e-01 -7.7528e-01
#  -7.7851e-01 -7.6296e-01 -7.2941e-01 -6.8649e-01 -6.4700e-01 -6.0135e-01
#  -5.1668e-01 -3.5461e-01 -1.2446e-01  1.0960e-01  2.7566e-01  3.4973e-01
#   3.6693e-01  3.6743e-01  3.6493e-01  3.5025e-01  3.1842e-01  2.7408e-01
#   2.2032e-01  1.5968e-01  9.2679e-02  2.4001e-02 -4.2606e-02 -1.0739e-01
#  -1.6991e-01 -2.2826e-01 -2.7407e-01 -3.1021e-01 -3.4458e-01 -3.8020e-01
#  -4.1561e-01 -4.4517e-01 -4.7684e-01 -5.0743e-01 -5.3101e-01 -5.4008e-01
#  -5.3941e-01 -5.3979e-01 -5.3551e-01 -5.1782e-01 -4.7450e-01 -4.0326e-01
#  -3.0969e-01 -2.1226e-01 -1.3415e-01 -7.9850e-02 -4.2934e-02 -1.4023e-02
#   6.3127e-03  1.3374e-02  1.9292e-03 -3.2878e-02 -8.5803e-02 -1.4717e-01
#  -2.0780e-01 -2.6970e-01 -3.3558e-01 -3.9781e-01 -4.4935e-01 -4.9013e-01
#  -5.2458e-01 -5.4962e-01 -5.6398e-01 -5.7559e-01 -5.8836e-01 -6.0254e-01
#  -6.1354e-01 -6.2503e-01 -6.3906e-01 -6.5384e-01 -6.6936e-01 -6.8688e-01
#  -7.1136e-01 -7.3746e-01 -7.5950e-01 -7.7499e-01 -7.8813e-01 -8.0179e-01
#  -8.1073e-01 -8.1522e-01 -8.1346e-01 -8.1020e-01 -8.0397e-01 -7.9797e-01
#  -7.9591e-01 -7.9603e-01 -8.0098e-01 -8.0830e-01 -8.1267e-01 -7.8676e-01
#  -7.0049e-01 -5.4492e-01 -3.4436e-01 -1.4717e-01  1.4400e-02  1.3525e-01
#   2.3174e-01  3.1293e-01  3.7401e-01  4.1251e-01  4.3057e-01  4.3704e-01
#   4.2535e-01  3.9101e-01  3.3585e-01  2.6265e-01  1.8310e-01  9.6799e-02
#   2.6423e-02 -2.7361e-02 -6.4349e-02 -8.7876e-02 -1.0000e-01 -8.9198e-02
#  -6.0574e-02  2.3727e-02  2.1472e-01  5.5736e-01  1.0339e+00  1.5689e+00
#   2.1137e+00  2.6502e+00  3.1621e+00  3.5754e+00  3.8176e+00  3.9004e+00
#   3.9005e+00  3.8876e+00  3.8793e+00  3.8756e+00  3.8731e+00  3.8524e+00
#   3.7751e+00  3.5853e+00  3.2514e+00  2.7859e+00  2.2418e+00  1.6779e+00
#   1.1336e+00  6.5429e-01  2.6315e-01 -2.5552e-02 -2.2418e-01 -3.5280e-01
#  -4.2527e-01 -4.6806e-01 -4.8383e-01 -4.8914e-01 -4.8363e-01 -4.7244e-01
#  -4.5977e-01 -4.4213e-01 -4.3210e-01 -4.2134e-01 -4.1137e-01 -3.9204e-01
#  -3.6658e-01 -3.4208e-01 -3.2095e-01 -3.0738e-01 -3.0061e-01 -3.0206e-01
#  -3.0844e-01 -3.2294e-01 -3.5098e-01 -3.8881e-01 -4.2600e-01 -4.5244e-01
#  -4.6749e-01 -4.7797e-01 -4.9486e-01 -5.2258e-01 -5.4985e-01 -5.6834e-01
#  -5.8237e-01 -6.0312e-01 -6.3103e-01 -6.5994e-01 -6.9270e-01 -7.3567e-01
#  -7.9194e-01 -8.5537e-01 -9.2301e-01 -9.8822e-01 -1.0334e+00 -1.0482e+00]
# filters = [-0.98249,-0.81463,-0.64004,-0.42288,-0.20597  -0.029173  0.1056
#   0.20445   0.28626   0.38637 ,  0.51516  , 0.6529 ,   0.80734 ,  0.98754
#   1.1807    1.3696    1.5262  ,  1.6299   , 1.6897 ,   1.7283  ,  1.78
#   1.8715    1.9949    2.1233  ,  2.2042   , 2.1756 ,   2.0161  ,  1.7627
#   1.5071    1.3128    1.1885  ,  1.1447   , 1.1804 ,   1.275   ,  1.4102
#   1.5652    1.7328    1.9321  ,  2.1884   , 2.4997 ,   2.7826  ,  2.9085
#   2.821     2.5587    2.205   ,  1.8158   , 1.4052 ,   1.0004  ,  0.62805
#   0.28964  -0.023883 -0.31325 , -0.55426  ,-0.72596,  -0.83837 , -0.89417
#  -0.90393  -0.88779  -0.85902 , -0.8353   ,-0.8094 ,  -0.77792 , -0.74127
#  -0.67809  -0.58831  -0.4813  , -0.35011  ,-0.17356,   0.05997 ,  ,0.32471
#   0.59459   0.86669   1.1249  ,  1.32     , 1.4029 ,   1.3627  ,  1.2236
#   1.019     0.78923   0.57543 ,  0.38622  , 0.19685,  -0.011919, -0.23279
#  -0.43755  -0.59553  -0.69374 , -0.74331  ,-0.77344,  -0.80224 , -0.82827
#  -0.85717  -0.89044  -0.93115 , -0.97726  ,-1.0168 ,  -1.0502  , -1.0797
#  -1.1004   -1.1109]

# dataset = "Adiac"
dataset = "50words"
# dataset = "Herring"
# dataset = "InlineSkate"
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

x = train_set_x[0]
print("input size: ", len(x))
# print("x: ", x)
filter_size = 4
full_filter = train_set_x[1]
filters = full_filter[:filter_size]
# filters = np.random.randn(filter_size)

repetitions = 1
exec_number = 1

b = np.array([0])

stride = 1

timings = []
errors = []

fraction = 0.99


def plot_signal(signal, title="signal"):
    plt.plot(range(0, len(signal)), signal)
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()


# filter_sizes = [value for value in range(1, len(full_filter))]
filter_sizes = [value for value in range(100, 101)]
for filter_size in filter_sizes:
    print("filter size: ", filter_size)
    filters = full_filter[:filter_size].copy()
    # x = zscore(x)
    # filters = zscore(filters)
    # plot_signal(x, "input signal")
    # plot_signal(filters, "znorm of the input filter in time domain")
    # print("filters: ", filters)
    # adjust padding
    mode = "full"
    if mode == "valid":
        pad = 0
    elif mode == "full":
        pad = len(filters) - 1
    energy_rate = 0.95
    conv_param = {'stride': stride, 'pad': pad, 'preserve_energy_rate': energy_rate}

    numpy_time, result_numpy = timeitrep(wrapper(np.correlate, x, filters, mode=mode), number=exec_number,
                                         repetition=repetitions)
    # plot_signal(result_numpy, "result numpy")

    numpy_time, result_john = timeitrep(wrapper(cross_corelate_john_compressed, x, filters, pad, energy_rate=energy_rate),
                                        number=exec_number,
                                        repetition=repetitions)

    print("rel error numpy john: ", rel_error(np.array(zscore(result_numpy), dtype=np.float32),
                                              np.array(zscore(result_john), dtype=np.float32)))
    print("rel error numpy john: ", rel_error(np.array(zscore(result_numpy), dtype=np.float64),
                                              np.array(zscore(result_john), dtype=np.float64)))

    print("abs error numpy john: ", abs_error(np.array(zscore(result_numpy), dtype=np.float64),
                                              np.array(zscore(result_john), dtype=np.float64)))

    conv_compressed, (result_compressed, _) = timeitrep(
        wrapper(conv_forward_fft_1D, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param),
        number=exec_number, repetition=repetitions)
    plot_signal(result_compressed[0, 0], "result compressed")

    print("rel error numpy fft: ", rel_error(np.array(zscore(result_numpy), dtype=np.float32),
                                             np.array(zscore(result_compressed[0, 0]), dtype=np.float32)))
    print("rel error numpy fft: ", rel_error(np.array(zscore(result_numpy), dtype=np.float64),
                                             np.array(zscore(result_compressed[0, 0]), dtype=np.float64)))

    print("abs error numpy fft: ", abs_error(np.array(zscore(result_numpy), dtype=np.float64),
                                              np.array(zscore(result_compressed[0, 0]), dtype=np.float64)))

    import matplotlib.pyplot as plt

    # plt.plot(range(0, len(xfft1)), np.abs(xfft1), color="red")
    plt.plot(range(0, len(result_compressed[0, 0])), result_compressed[0, 0], color="blue")
    plt.plot(range(0, len(result_numpy)), result_numpy, color="green")
    plt.title("cross-correlation output: numpy (green) vs. fft (blue)")
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.show()

    sys.exit()

    # W = len(x)
    # WW = len(filters)
    # fftsize = next_power2(W + WW - 1)
    # padded_x = (np.pad(x, (pad, pad), 'constant'))
    # out_W = W + 2 * pad - WW + 1
    # preserve_energy_rate = 1.0
    # conv_kshape, result_ncc = timeitrep(
    #     wrapper(convolve1D_fft_scale, x, filters, fftsize, out_size=out_W,
    #             preserve_energy_rate=preserve_energy_rate), number=exec_number, repetition=repetitions)
    #
    # plot_signal(result_ncc, "our fft")

    # conv_kshape, result_ncc = timeitrep(
    #     wrapper(original_ncc_c, x, filters), number=exec_number, repetition=repetitions)
    #
    # plot_signal(result_ncc, "result ncc")

    # conv_compressed, (result_compressed, _) = timeitrep(
    #     wrapper(conv_forward_fft_1D_compress_compare, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param,
    #             preserve_energy_rate=1.0),
    #     number=exec_number, repetition=repetitions)

    # conv_compressed, (result_compressed, _) = timeitrep(
    #     wrapper(conv_forward_fft_1D_compress_energy, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param,
    #             energy_rate=0.9985),
    #     number=exec_number, repetition=repetitions)

    # conv_compressed, (result_compressed, _) = timeitrep(
    #     wrapper(conv_forward_fft_1D_compress_perf, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param,
    #             compress_rate=100),
    #     number=exec_number, repetition=repetitions)

    conv_naive_time, (result_naive, _) = timeitrep(
        wrapper(conv_forward_naive_1D, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param),
        number=exec_number, repetition=repetitions)

    xfft2 = result_compressed[0][0]
    xfft3 = result_naive[0][0]

    # print("xfft2/xfft3: ", xfft2/xfft3)

    print("abs error naive compressed raw: ", abs_error(xfft2, xfft3))

    from scipy.stats.mstats import zscore

    # xfft2 = zscore(xfft2)
    # xfft3 = zscore(xfft3)

    # import matplotlib.pyplot as plt
    #
    # # plt.plot(range(0, len(xfft1)), np.abs(xfft1), color="red")
    # plt.plot(range(0, len(xfft2)), xfft2, color="blue")
    # plt.plot(range(0, len(xfft3)), xfft3, color="green")
    # plt.title("cross-correlation output: naive (green) vs. compressed (blue)")
    # plt.xlabel('time')
    # plt.ylabel('Amplitude')
    # plt.show()
    #
    # print("abs error naive compressed after zscore: ", abs_error(xfft2, xfft3))

    conv_kshape, result_john = timeitrep(
        wrapper(cross_corelate_john, x, filters, padding), number=exec_number, repetition=repetitions)

    conv_fft_time, result_adam = timeitrep(
        wrapper(cross_correlate_adam, x, filters, padding), number=exec_number, repetition=repetitions)

    print("abs error john adam: ", abs_error(result_john, result_adam))

    conv_kshape, result_kshape = timeitrep(
        wrapper(cross_correlate, x, filters), number=exec_number, repetition=repetitions)

    conv_naive_time, (result_naive, _) = timeitrep(
        wrapper(conv_forward_naive_1D, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param),
        number=exec_number, repetition=repetitions)

    for fft_back in range(0, 1):
        reshaped_x = reshape_3d_rest(x)
    reshaped_filters = reshape_3d_rest(filters)
    conv_fft_time_compressed, (result_fft_compressed, _) = timeitrep(
        wrapper(conv_forward_fft_1D_compress, reshaped_x, reshaped_filters, b, conv_param,
                fft_back=fft_back, index_back=None),
        number=exec_number, repetition=repetitions)
    print("compress_rate: ", fft_back, "error: ", abs_error(result_naive, result_fft_compressed))

    conv_kshape, result_kshape = timeitrep(
        wrapper(cross_correlate_test, x, filters), number=exec_number, repetition=repetitions)

    conv_kshape, result_kshape = timeitrep(
        wrapper(original_ncc_c, x, filters), number=exec_number, repetition=repetitions)

    conv_fft_time_compressed_fraction, (result_fft_compressed_fraction, _) = timeitrep(
        wrapper(conv_forward_fft_1D_compress_fraction, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param),
        number=exec_number, repetition=repetitions)

    timings.append(conv_fft_time_compressed_fraction / conv_fft_time)
    errors.append(abs_error(result_fft, result_fft_compressed_fraction))

    print("filter sizes: ", filter_sizes)
    print("timings: ", timings)
    print("absolute errors: ", errors)
    print("size of time series: ", len(x))

    df_input = {'filter-sizes': filter_sizes, 'timings': timings, 'errors': errors}
    df = DataFrame(data=df_input)
    # df.set_index('preserve-energy', inplace=True)

    fig = plt.figure()  # create matplot figure
    ax = fig.add_subplot(111)  # create matplotlib axes
    ax2 = ax.twinx()  # create another axes that shares the same x-axis as ax

    width = 0.4

    df.timings.plot(kind='bar', color='red', ax=ax)
    df.errors.plot(kind='line', color='blue', ax=ax2)

    left_ylabel = "Relative execution time to pure adamfft method"
    ax.set_ylabel(left_ylabel)
    right_ylabel = "Absolute error"
    ax2.set_ylabel(right_ylabel)
    plt.title("Dataset: " + dataset + "\n(preserved energy vs relative execution time to the full fft convolution "
              + "& absolute error, repetitions: " + str(repetitions) + ")")
    ax.set_xticklabels(filter_sizes)
    ax.set_xlabel("Size of the filter")
    red_patch = mpatches.Patch(color='red', label=left_ylabel)
    blue_patch = mpatches.Patch(color='blue', label=right_ylabel)
    plt.legend(handles=[red_patch, blue_patch], loc='upper right', ncol=1,
               borderaxespad=0.0)
    plt.show()
