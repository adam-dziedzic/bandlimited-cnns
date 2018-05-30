from cs231n.gradient_check import eval_numerical_gradient_array
from cs231n.layers import fft_pool_forward_1D, fft_pool_backward_1D
from cs231n.utils.general_utils import *

np.random.seed(231)
channel = 1
nr_filters = 1
nr_data = 1
W = 8

x = np.random.randn(W)
x = x.reshape(1, 1, -1)

pool_width = 5
pool_stride = 3
pool_param = {'pool_width': pool_width, 'stride': pool_stride}
out_pool = np.int(((W - pool_width) // pool_stride) + 1)
print("out_pool: ", out_pool)

dout = np.array([[np.random.randn(out_pool)]])
# print("dout: ", dout)

out, cache = fft_pool_forward_1D(x, pool_param)
# print("out: ", out)
dx = fft_pool_backward_1D(dout, cache)

# print("dx: ", dx)

dx_num = eval_numerical_gradient_array(lambda x: fft_pool_forward_1D(x, pool_param)[0], x, dout)

plot_signal(dx[0, 0], "obtained error")
plot_signal(dx_num[0, 0], "expected numerical error")
plot_signals(dx_num[0, 0], dx[0, 0], label_x="expected dx", label_y="obtained dx")
print('Testing naive conv_relu_pool')
print('dx error: ', rel_error(dx_num[0, 0], dx[0, 0]))
print('abs error: ', abs_error(dx_num[0, 0], dx[0, 0]))
