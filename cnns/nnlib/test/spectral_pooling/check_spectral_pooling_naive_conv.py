from nnlib.gradient_check import eval_numerical_gradient_array
from nnlib.layer_utils import *
from nnlib.utils.general_utils import *

"""
Spectral pooling and naive convolution.
"""

current_file_name = __file__.split("/")[-1].split(".")[0]
print("current file name: ", current_file_name)

np.random.seed(231)
nr_data = 1
nr_filters = 1
channel = 1
W = 128
WW = 8

x = np.random.randn(nr_data, channel, W)
w = np.random.randn(nr_filters, channel, WW)
b = np.random.randn(nr_filters, )

stride = 1
pad = 0
conv_param = {'stride': stride, 'pad': pad}
out_conv = np.int(((W + 2 * pad - WW) // stride) + 1)
# print("out_conv: ", out_conv)

pool_width = 5
pool_stride = 2
pool_param = {'pool_width': pool_width, 'stride': pool_stride}
out_pool_size = np.int(((out_conv - pool_width) // pool_stride) + 1)
print("out pool size: ", out_pool_size)

out_pool, cache = conv_relu_pool_fft_forward_naive_1D(x, w, b, conv_param, pool_param)
# print("out numpy: ", out_numpy)
# dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)
plot_signal(out_pool[0, 0], "signal after applying spectral pooling")
# dout = out_pool
# dout = np.random.randn(nr_data, nr_filters, out_pool_size)
dout = np.ones((1, 1, out_pool_size))
dx, dw, db = conv_relu_pool_fft_backward_naive_1D(dout, cache)

# print("dx numpy conv pool fft: ", dx)
# print("dw numpy conv pool fft: ", dw)

dx_num = eval_numerical_gradient_array(
    lambda x: conv_relu_pool_fft_forward_naive_1D(x, w, b, conv_param, pool_param)[0], x,
    out_pool)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_relu_pool_fft_forward_naive_1D(x, w, b, conv_param, pool_param)[0], w,
    out_pool)
db_num = eval_numerical_gradient_array(
    lambda b: conv_relu_pool_fft_forward_naive_1D(x, w, b, conv_param, pool_param)[0], b,
    out_pool)

print('Testing naive conv_relu_fft pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

plot_signals(dx_num[0, 0], dx[0, 0], label_x="expected num dx", label_y="obtained dx")

"""
Check the same just for the naive convolution.
"""

out_pool, cache = conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)
# print("out numpy: ", out_numpy)
# dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)
plot_signal(out_pool[0, 0], "signal after applying spectral pooling")
# dout = out_pool
dout = np.random.randn(nr_data, nr_filters, out_pool_size)
# dout = np.ones((1, 1, out_pool_size))
dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)

# print("dx numpy conv pool fft: ", dx)
# print("dw numpy conv pool fft: ", dw)

dx_num = eval_numerical_gradient_array(
    lambda x: conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)[0], x,
    out_pool)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)[0], w,
    out_pool)
db_num = eval_numerical_gradient_array(
    lambda b: conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)[0], b,
    out_pool)

print('Testing naive conv_relu_pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

plot_signals(dx_num[0, 0], dx[0, 0], label_x="expected num dx", label_y="obtained dx")

