from cs231n.layers import fft_pool_forward_1D, fft_pool_backward_1D
from cs231n.load_time_series import load_data
from cs231n.utils.general_utils import *

# print(__name__)
current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)

np.random.seed(231)
channel = 1
nr_filters = 1
nr_data = 1
W = 512
WW = 100

# dataset = "Adiac"
dataset = "50words"
# dataset = "Herring"
# dataset = "InlineSkate"
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]

x = train_set_x[0]
x = np.array(x, dtype=np.float64)
W = len(x)

w = train_set_x[1]
w = np.array(w, dtype=np.float64)
WW = len(w)

# x = np.random.randn(W)
# w = np.random.randn(nr_filters, channel, WW)
b = np.random.randn(nr_filters, )

x = x.reshape(nr_data, channel, -1)
w = w.reshape(nr_data, channel, -1)
b = b.reshape(1, )

pool_width = 1
pool_stride = 1
pool_param = {'pool_width': pool_width, 'stride': pool_stride}
out_pool = np.int(((W - pool_width) // pool_stride) + 1)
# out_pool = W
print("out_pool: ", out_pool)

# dout = np.array([[np.random.randn(out_pool)]])
# print("dout: ", dout)

plot_signal(x[0, 0], "input signal")
out, cache = fft_pool_forward_1D(x, pool_param)
# print("out: ", out)
dout = out
plot_signal(dout[0, 0], "the gradient")
dx = fft_pool_backward_1D(dout, cache)
plot_signal(dx[0, 0], "dx output")
plot_signals(x[0, 0], dx[0, 0], label_x="input", label_y="output", linestyle="dotted")
print('dx error: ', rel_error(dx[0, 0], x[0, 0]))
print('abs error: ', abs_error(dx[0, 0], x[0, 0]))

# print("dx: ", dx)
# dx_num = eval_numerical_gradient_array(lambda x: fft_pool_forward_1D(x, pool_param)[0], x, dout)


# plot_signal(dx[0, 0], "obtained error")
# plot_signal(dx_num[0, 0], "expected numerical error")
# plot_signals(dx_num[0, 0], dx[0, 0], label_x="expected dx", label_y="obtained dx")
# print('Testing spectral_pool')
# print('dx error: ', rel_error(dx_num[0, 0], dx[0, 0]))
# print('abs error: ', abs_error(dx_num[0, 0], dx[0, 0]))
