import numpy as np

from nnlib.gradient_check import eval_numerical_gradient_array
from nnlib.layers import fft_pool_forward_1D, fft_pool_backward_1D
from nnlib.utils.general_utils import rel_error

np.random.seed(231)
channel = 1
nr_filters = 1
nr_data = 1
W = 8

x = np.array([[[1.0, 2, 3, 4, 5, 6, 7, 8]]])
x = x.reshape(1, 1, -1)

pool_width = 3
pool_stride = 3
pool_param = {'pool_width': pool_width, 'stride': pool_stride}
out_pool = np.int(((W - pool_width) // pool_stride) + 1)
print("out_pool: ", out_pool)

dout = np.array([[np.ones(out_pool)]])
print("dout: ", dout)

out, cache = fft_pool_forward_1D(x, pool_param)
print("out: ", out)
dx = fft_pool_backward_1D(dout, cache)

print("dx: ", dx)

dx_num = eval_numerical_gradient_array(lambda x: fft_pool_forward_1D(x, pool_param)[0], x, dout)

print('Testing naive conv_relu_pool')
print('dx error: ', rel_error(dx_num, dx))
