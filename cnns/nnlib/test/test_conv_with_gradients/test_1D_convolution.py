# 1D naive convolution - different stride and padding tested
from nnlib.gradient_check import eval_numerical_gradient_array
from nnlib.layer_utils import *
from nnlib.utils.general_utils import abs_error, rel_error

np.random.seed(231)
channel = 1
nr_filters = 1
nr_data = 1
# WW = 5
# W = 10
WW = 49
W = 1024

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

print('spectral pool: testing numpy conv relu spectral (fft based) pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

"""
Going back to numpy convolution, relu and spectral (fft) pooling.
"""

out_numpy_conv_spectral_pool, cache = conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)
# print("out numpy: ", out_numpy)
# dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)
dx, dw, db = conv_relu_pool_fft_backward_numpy_1D(dout, cache)
are_close = np.allclose(out_numpy_conv_spectral_pool, out_numpy_conv_spectral_pool)
print("are outputs of naive and naive conv fft pool close: ", are_close)

# print("dx numpy conv pool fft: ", dx)
# print("dw numpy conv pool fft: ", dw)

dx_num = eval_numerical_gradient_array(
    lambda x: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], w, dout)
db_num = eval_numerical_gradient_array(
    lambda b: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], b, dout)

print('Testing numpy conv reul and spectral (fft based) pooling')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))


"""
ExperimentSpectralSpatial the naive convolution.
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

"""
ExperimentSpectralSpatial the numpy convolution
"""
out_numpy, cache = conv_relu_pool_forward_numpy_1D(x, w, b, conv_param, pool_param)
# print("out numpy: ", out_numpy)
# dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)
dx, dw, db = conv_relu_pool_backward_numpy_1D(dout, cache)

are_close = np.allclose(out_naive, out_numpy)
print("are outputs of naive and numpy close: ", are_close)
assert are_close
# print("dx numpy: ", dx)
# print("dw numpy: ", dw)

dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], x,
                                       dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], w,
                                       dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], b,
                                       dout)

print('Testing numpy conv_relu_pool for numerical gradients')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

# print("dx numpy: ", dx)
# print("dw numpy: ", dw)

"""
FFT based convolution
"""
out_fft, cache = conv_relu_pool_forward_fft_1D(x, w, b, conv_param, pool_param)
# print("out fft: ", out_fft)
dx, dw, db = conv_relu_pool_backward_fft_1D(dout, cache)

are_close = np.allclose(out_naive, out_fft)
print("are outputs of naive and fft close: ", are_close)

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

# print("dx fft: ", dx)
# print("dx num: ", dx_num)
# print("dw fft: ", dw)
# print("dw num: ", dw)

print('dx abs error: ', abs_error(dx_num, dx))

out_numpy_conv_fft_pool, cache = conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)
# print("out numpy: ", out_numpy)
# dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)
dx, dw, db = conv_relu_pool_fft_backward_numpy_1D(dout, cache)
are_close = np.allclose(out_naive, out_numpy_conv_fft_pool)
print("are outputs of naive and numpy conv fft pool close: ", are_close)

# print("dx numpy conv pool fft: ", dx)
# print("dw numpy conv pool fft: ", dw)

dx_num = eval_numerical_gradient_array(
    lambda x: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], x,
    dout)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], w,
    dout)
db_num = eval_numerical_gradient_array(
    lambda b: conv_relu_pool_fft_forward_numpy_1D(x, w, b, conv_param, pool_param)[0], b,
    dout)

print('Testing numpy conv_relu_fft pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

"""
Spectral pooling and naive convolution.
"""

out_naive_conv_fft_pool, cache = conv_relu_pool_fft_forward_naive_1D(x, w, b, conv_param, pool_param)
# print("out numpy: ", out_numpy)
# dx, dw, db = conv_relu_pool_backward_naive_1D(dout, cache)
dx, dw, db = conv_relu_pool_fft_backward_naive_1D(dout, cache)
are_close = np.allclose(out_naive, out_naive_conv_fft_pool)
print("are outputs of naive and naive conv fft pool close: ", are_close)

# print("dx numpy conv pool fft: ", dx)
# print("dw numpy conv pool fft: ", dw)

dx_num = eval_numerical_gradient_array(
    lambda x: conv_relu_pool_fft_forward_naive_1D(x, w, b, conv_param, pool_param)[0], x,
    dout)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_relu_pool_fft_forward_naive_1D(x, w, b, conv_param, pool_param)[0], w,
    dout)
db_num = eval_numerical_gradient_array(
    lambda b: conv_relu_pool_fft_forward_naive_1D(x, w, b, conv_param, pool_param)[0], b,
    dout)

print('Testing numpy conv_relu_fft pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))
