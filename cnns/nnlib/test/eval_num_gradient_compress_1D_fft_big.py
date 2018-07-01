from nnlib.gradient_check import eval_numerical_gradient_array
from nnlib.layer_utils import *
from nnlib.layers import *
from nnlib.load_time_series import load_data
from nnlib.utils.general_utils import *

np.random.seed(231)
channel = 1
nr_filters = 1
nr_data = 1
W = 1024
WW = 49  # 49

# dataset = "Adiac"
# dataset = "Herring"
# dataset = "InlineSkate"

# dataset = "50words"
# datasets = load_data(dataset)
# train_set_x, train_set_y = datasets[0]
# valid_set_x, valid_set_y = datasets[1]
#
# x = train_set_x[0]
# x = np.array(x, dtype=np.float64)
# W = len(x)
#
# w = train_set_x[1]
# w = np.array(w, dtype=np.float64)
# WW = len(w)

print("W: ", W)
print("WW: ", WW)

x = np.random.randn(W)
w = np.random.randn(WW)
w = np.pad(w, (0, W-WW), "constant")
print("length of w: ", len(w))
# print("w: ", w)
# w = np.random.randn(nr_filters, channel, WW)
b = np.random.randn(nr_filters, )

x = x.reshape(nr_data, channel, -1)
w = w.reshape(nr_filters, channel, -1)
b = b.reshape(nr_filters, )

pool_width = 5
pool_stride = 2
pool_param = {'pool_width': pool_width, 'stride': pool_stride}
out_pool = np.int(((W - pool_width) // pool_stride) + 1)
print("out_pool: ", out_pool)

dout = np.array([[np.random.randn(out_pool)]])
# print("dout: ", dout)

# plot_signal(x[0, 0], "input signal")
out, cache = fft_pool_forward_1D(x, pool_param)
# print("out: ", out)
# plot_signal(dout[0, 0], "the gradient")
dx = fft_pool_backward_1D(dout, cache)

# print("dx: ", dx)

dx_num = eval_numerical_gradient_array(lambda x: fft_pool_forward_1D(x, pool_param)[0], x, dout)

# plot_signal(dx[0, 0], "obtained error")
# plot_signal(dx_num[0, 0], "expected numerical error")
# plot_signals(dx_num[0, 0], dx[0, 0], label_x="expected dx", label_y="obtained dx")
print('Testing fft pool forward 1D')
print('dx error: ', rel_error(dx_num[0, 0], dx[0, 0]))
print('abs error: ', abs_error(dx_num[0, 0], dx[0, 0]))

"""
Verify the gradients for the convolution
"""
conv_stride = 1
conv_pad = 2
conv_param = {'stride': conv_stride, 'pad': conv_pad}
out_conv = np.int(((W + 2 * conv_pad - WW) // conv_stride) + 1)
# print("out_conv: ", out_conv)

pool_width = 5
pool_stride = 2
pool_param = {'pool_width': pool_width, 'stride': pool_stride}
out_pool = np.int(((out_conv - pool_width) // pool_stride) + 1)
# print("out_pool: ", out_pool)

dout = np.random.randn(nr_data, nr_filters, out_pool)
# dout = np.array([[np.ones(out_pool)]])
# print("dout: ", dout)

"""
spectral pool
"""

out_pool_fft, cache = conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)
# print("out naive: ", out_naive)
dx, dw, db = conv_relu_pool_fft_backward_numpy_1D(dout, cache)

# print("dx naive: ", dx)
# print("dw naive: ", dw)

dx_num = eval_numerical_gradient_array(
    lambda x: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], w, dout)
db_num = eval_numerical_gradient_array(
    lambda b: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], b, dout)

print('Testing numpy conv_relu_spectral (fft based) pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

"""
max pool
"""

out_naive, cache = conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)
# print("out naive: ", out_naive)
dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)

# print("dx naive: ", dx)
# print("dw naive: ", dw)

dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)[0], x,
                                       dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)[0], w,
                                       dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)[0], b,
                                       dout)

print('Testing naive conv_relu_pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))
