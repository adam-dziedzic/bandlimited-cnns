import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os
from cnns.nnlib.utils.general_utils import get_log_time
import numpy as np

# http://ksrowell.com/blog-visualizing-data/2012/02/02/optimal-colors-for-graphs/
MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)

legend_position = 'lower left'
frameon = False
bbox_to_anchor = (0.0, -0.1)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


lw = 2  # the line width
ylabel_size = 25
legend_size = 20
font = {'size': 30}
title_size = 25
matplotlib.rc('font', **font)

markers = ["+", "o", "v", "s", "D", "^", "+", 'o', 'v', '+', 'v', 'D', '^', '+']
linestyles = [":", "-", "--", ":", "-", "--", "-", "--", ':', ':', "-", "-",
              "-"]

dir_path = os.path.dirname(os.path.realpath(__file__))

fig = plt.figure(figsize=(15, 7))

plt.subplot(2, 1, 1)

plt.title("LeNet on MNIST", fontsize=title_size)

# FFT based compression for every convolutional layer.
static_x = (
    0, 8.3855401, 11.87663269, 14.5, 17.8, 21.0941708, 24, 26, 29.26308507,
    39.203635, 49, 53.53477122, 67.50730502, 77.77388775
)
static_y = (
    93.69, 93.42, 93.12, 93, 93.06, 93.24, 93.16, 93.2, 92.89, 92.61, 91.95,
    91.64, 89.71, 87.97
)

mix_x = (0,
         20,
         28.59,
         48.24,
         71,
         79)
mix_y = (93.69,
         92.73,
         91.99,
         88.85,
         81.71,
         74.33)

energy_x = (
    0, 0.048475708, 1.209367931, 4.962953113, 12.35137482, 18.63192462,
    31.01018447,
    39.92, 50.4, 75.84286825)
energy_y = (
    93.69, 93.69, 93.12, 93.01, 92.39, 91.32, 88.99, 87.97, 83.84, 69.47)

svd_x = (
    0,
    11,
    23,
    30,
    42,
    49,
    61,
    74,
    80,
    93
)

svd_y = (
    93.73,
    93.36,
    92.91,
    92.53,
    92.02,
    91.4,
    89.7,
    85.11,
    80.91,
    10
)

fft_preprocess_x = [
    0,
    5,
    15,
    25,
    35,
    45,
    50,
    60,
    75,
    80,
    90,
    95
]

fft_preprocess_y = [
    93.73,
    93.63,
    93.45,
    93.3,
    93.02,
    92.71,
    92.69,
    91.9,
    90.68,
    89.34,
    82.98,
    78.07
]

train_on_svd_x = [
    0,
    1,
    25,
    50,
    75
]

train_on_svd_y = [
    93.73,
    64.23,
    61.88,
    64.15,
    57.45,
]

train_on_svd_x_p_by_p = [
    0,
    20,
    40,
    60,
    80,
]

train_on_svd_y_p_by_p = [
    93.73,
    41.8,
    38.32,
    35.94,
    30.52,
]

file_name = 'sum_singular_values_index_1_counter_local_surface.txt'
sst = np.genfromtxt(file_name, delimiter=';')
x_1 = sst[:, 0]
y_1 = sst[:, 2]

plt.plot(x_1, y_1,
         # label='train on SVD representation (3 channels, V*D*U is p by p)',
         label="1st conv layer",
         lw=lw,
         linestyle='--',
         marker='',
         color=get_color(MY_BLUE)
         )

file_name = 'sum_singular_values_index_2_counter_3_local_surface.txt'
sst2 = np.genfromtxt(file_name, delimiter=';')
limit = 3000
x_2 = sst2[:limit, 1]
y_2 = sst2[:limit, 2]

plt.plot(x_2, y_2,
         # label='train on SVD representation (3 channels, V*D*U is p by p)',
         label="2nd conv layer",
         lw=lw,
         linestyle='-',
         marker='o',
         color=get_color(MY_RED)
         )

plt.xlabel("Training Iteration")
plt.ylabel("Sum of $\sigma_i$'s", fontsize=ylabel_size)
plt.ylim(0, 110000)
plt.xlim(0, limit)
# plt.grid()
plt.legend(loc=legend_position,
           frameon=frameon,
           prop={'size': legend_size},
           bbox_to_anchor=bbox_to_anchor,
           ncol=2,
           )

fig.tight_layout()
# plt.show()
format = "pdf"  # "png" or "pdf"
file_name = dir_path + "/" + "mnist_iterations_singular_values_sum"
file_name += "_" + get_log_time() + "." + format
fig.savefig(file_name,
            bbox_inches='tight', transparent=True)
plt.close()
