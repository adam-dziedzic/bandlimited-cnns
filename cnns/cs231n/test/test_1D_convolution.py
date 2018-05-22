# 1D naive convolution - different stride and padding tested
from cs231n.gradient_check import eval_numerical_gradient_array
from cs231n.layer_utils import *
from cs231n.utils.general_utils import abs_error


def rel_error(x, y):
    """ returns relative error """
    # print("np.abs(x-y): ", np.abs(x-y))
    # print("np.abs(x) + np.abs(y): ", np.abs(x) + np.abs(y))
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


np.random.seed(231)
channel = 1
nr_filters = 1
nr_data = 1
WW = 5
W = 10

x = np.random.randn(nr_data, channel, W)
w = np.random.randn(nr_filters, channel, WW)
b = np.random.randn(nr_filters, )

# x = np.array([[[1.0, 2, 3, 4, 5]]])
# w = np.array([[[1.0, 2]]])
# b = np.array([0.0])

stride = 1
pad = 2
conv_param = {'stride': stride, 'pad': pad}
out_conv = np.int(((W + 2 * pad - WW) // stride) + 1)
print("out_conv: ", out_conv)

pool_width = 2
pool_stride = 3
pool_param = {'pool_width': pool_width, 'stride': pool_stride}
out_pool = np.int(((out_conv - pool_width) // pool_stride) + 1)
print("out_pool: ", out_pool)

dout = np.random.randn(nr_data, nr_filters, out_pool)
# dout = np.array([[np.ones(out_pool)]])
print("dout: ", dout)

out_naive, cache = conv_relu_pool_forward_naive_1D(x, w, b, conv_param, pool_param)
print("out naive: ", out_naive)
dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)

print("dx naive: ", dx)
print("dw naive: ", dw)

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

out_numpy, cache = conv_relu_pool_forward_numpy_1D(x, w, b, conv_param, pool_param)
print("out numpy: ", out_numpy)
# dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)
dx, dw, db = conv_relu_pool_backward_numpy_1D(dout, cache)

are_close = np.allclose(out_naive, out_numpy)
print("are outputs of naive and numpy close: ", are_close)
assert are_close

print("dx numpy: ", dx)
print("dw numpy: ", dw)

dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], x,
                                       dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], w,
                                       dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], b,
                                       dout)

print('Testing numpy conv_relu_pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

print('Testing numpy conv_relu_pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

print("dx numpy: ", dx)
print("dw numpy: ", dw)

out_fft, cache = conv_relu_pool_forward_fft_1D(x, w, b, conv_param, pool_param)
print("out fft: ", out_fft)
dx, dw, db = conv_relu_pool_backward_fft_1D(dout, cache)

are_close = np.allclose(out_naive, out_fft)
print("are outputs of naive and fft close: ", are_close)
assert are_close

dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward_fft_1D(x, w, b, conv_param, pool_param)[0], x,
                                       dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward_fft_1D(x, w, b, conv_param, pool_param)[0], w,
                                       dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward_fft_1D(x, w, b, conv_param, pool_param)[0], b,
                                       dout)

print('Testing fft conv_relu_pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

print("dx fft: ", dx)
print("dx num: ", dx_num)
print("dw fft: ", dw)
print("dw num: ", dw)

print('dx abs error: ', abs_error(dx_num, dx))
