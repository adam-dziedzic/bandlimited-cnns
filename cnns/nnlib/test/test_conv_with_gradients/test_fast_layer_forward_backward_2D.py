from nnlib.fast_layers import conv_forward_fast, conv_backward_fast
import numpy as np

np.random.seed(231)
x = np.random.randn(100, 3, 31, 31)
w = np.random.randn(25, 3, 3, 3)
b = np.random.randn(25,)
dout = np.random.randn(100, 25, 16, 16)
#conv_param = {'stride': 2, 'pad': 1}
conv_param = {'stride': 1, 'pad': 0}

out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)