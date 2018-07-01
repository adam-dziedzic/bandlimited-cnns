import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pandas import DataFrame
from nnlib.data_utils import get_CIFAR10_data
from nnlib.layers import *
from nnlib.load_time_series import load_data
from nnlib.utils.general_utils import reshape_3d_rest, abs_error
from nnlib.utils.perf_timing import wrapper, timeitrep

np.random.seed(231)

# dataset = "Adiac"
dataset = "50words"
# dataset = "Herring"
# dataset = "InlineSkate"
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

data = get_CIFAR10_data(cifar10_dir='../datasets/cifar-10-batches-py')
for k, v in data.items():
    print('%s: ' % k, v.shape)

num_train = 3000
num_valid = 300

small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'][:num_valid],
    'y_val': data['y_val'][:num_valid],
}
#print("X_train: ", small_data['X_train'].shape)
#print("y_train: ", data['y_train'])

small_data['X_train'] = small_data['X_train'].reshape(
    small_data['X_train'].shape[0], small_data['X_train'].shape[1], -1)
#print("x_train shape: ", small_data['X_train'].shape)

x = small_data['X_train'][0][0]

# x = train_set_x[0]
# x = np.random.rand(4096)
print("input signal size: ", len(x))
filter_size = 4
full_filter = train_set_x[1]
filters = full_filter[:filter_size]
# filters = np.random.randn(filter_size)

repetitions = 1
exec_number = 1

b = np.array([0])

#rates_raw = [1.0, 0.99999, 0.9999, 0.9998, 0.9995, 0.999, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.90, 0.80,
             # 0.70, 0.60, 0.50, 0.40,
             # 0.30, 0.20, 0.10, 0.05]
# rates_raw = [0.98, 0.30, 0.20, 0.10, 0.05]
rates_raw = [0.99]
rates = np.array(rates_raw)

stride = 1

mode = "full"
if mode == "valid":
    padding = 0
elif mode == "full":
    padding = len(filters) - 1
conv_param = {'stride': stride, 'pad': padding, 'preserve_energy_rate': 1.0}

timings = []
errors = []


conv_naive_time, (result_naive, _) = timeitrep(
    wrapper(conv_forward_naive_1D, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param),
    number=exec_number, repetition=repetitions)

for rate in rates:
    conv_param['preserve_energy_rate'] = rate
    print("input filter size: ", len(filters))
    conv_fft_time_compressed, (result_fft, _) = timeitrep(
        wrapper(conv_forward_fft_1D, reshape_3d_rest(x), reshape_3d_rest(filters), b, conv_param),
        number=exec_number, repetition=repetitions)
    timings.append(conv_fft_time_compressed)
    errors.append(abs_error(result_naive, result_fft))

print("timings: ", timings)
print("absolute errors: ", errors)
print("size of time series: ", len(x))

df_input = {'preserve-energy': rates, 'timings': timings, 'errors': errors}
df = DataFrame(data=df_input)
# df.set_index('preserve-energy', inplace=True)

fig = plt.figure()  # create matplot figure
ax = fig.add_subplot(111)  # create matplotlib axes
ax2 = ax.twinx()  # create another axes that shares the same x-axis as ax

width = 0.4

df.timings.plot(kind='bar', color='red', ax=ax)
df.errors.plot(kind='line', color='blue', ax=ax2)

ax.set_ylabel("Execution time (sec)")
ax2.set_ylabel("Absolute error")
plt.title("Dataset: " + dataset + "\n(preserved energy vs execution time & absolute error, repetitions: " + str(
    repetitions) + ")")
ax.set_xticklabels(rates_raw)
ax.set_xlabel("Preserved energy of time-series")
red_patch = mpatches.Patch(color='red', label='Execution time (sec)')
blue_patch = mpatches.Patch(color='blue', label='Absolute error')
plt.legend(handles=[red_patch, blue_patch], loc='upper right', ncol=1,
           borderaxespad=0.0)
plt.show()
