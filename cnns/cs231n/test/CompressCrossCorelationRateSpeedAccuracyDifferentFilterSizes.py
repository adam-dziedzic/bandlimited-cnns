import matplotlib.patches as mpatches
from pandas import DataFrame

from cs231n.layers import *
from cs231n.layers import _ncc_c
from cs231n.load_time_series import load_data
from cs231n.utils.general_utils import reshape_3d_rest, abs_error
from cs231n.utils.perf_timing import wrapper, timeitrep

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
print("input size: ", len(x))
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

# filter_sizes = [value for value in range(1, len(full_filter))]
filter_sizes = [value for value in range(100, 101)]
for filter_size in filter_sizes:
    print("filter size: ", filter_size)
    filters = full_filter[:filter_size].copy()
    # adjust padding
    mode = "full"
    if mode == "valid":
        padding = 0
    elif mode == "full":
        padding = len(filters) - 1
    conv_param = {'stride': stride, 'pad': padding}

    conv_naive_time, (result_naive, _) = timeitrep(

        wrapper(conv_forward_naive_1D, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param),
        number=exec_number, repetition=repetitions)
    #
    # conv_fft_time, (result_fft, _) = timeitrep(
    #     wrapper(conv_forward_fft_1D, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param),
    #     number=exec_number, repetition=repetitions)

    # conv_kshape, result_kshape = timeitrep(
    #     wrapper(cross_correlate, x, filters), number=exec_number, repetition=repetitions)
    for fft_back in range(0, 1):
        reshaped_x = reshape_3d_rest(x)
        reshaped_filters = reshape_3d_rest(filters)
        conv_fft_time_compressed, (result_fft_compressed, _) = timeitrep(
            wrapper(conv_forward_fft_1D_compress, reshaped_x, reshaped_filters, b, conv_param,
                    fft_back=fft_back, index_back=None),
            number=exec_number, repetition=repetitions)
        print("index_back: ", fft_back, "error: ", abs_error(result_naive, result_fft_compressed))

    conv_kshape, result_kshape = timeitrep(
        wrapper(cross_correlate_test, x, filters), number=exec_number, repetition=repetitions)

    conv_kshape, result_kshape = timeitrep(
        wrapper(_ncc_c, x, filters), number=exec_number, repetition=repetitions)

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
