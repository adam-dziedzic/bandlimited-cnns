import numpy as np
from numpy.fft import fft

from cs231n.load_time_series import load_data
from cs231n.utils.general_utils import abs_error

np.random.seed(231)

x = [1]
print("x: ", x)
print("fft x:", fft(x))

x0 = [1, 2]
print("x0: ", x0)
print("fft x0:", fft(x0))

x1 = [1, 2, 3]
print("x1: ", x1)
print("fft x1:", fft(x1))

x2 = [2, 2, 3, 4]
print("x2: ", x2)
print("fft x2:", fft(x2))

x3 = [1, 2, 3, 4, 5]
print("x3: ", x3)
print("fft x3:", fft(x3))

x3 = [1, 2, 3, 4, 5, 6]
print("x3: ", x3)
print("fft x3:", fft(x3))

x3 = [1, 2, 3, 4, 5, 6, 7]
print("x3: ", x3)
print("fft x3:", fft(x3))

x3 = [1, 2, 3, 4, 5, 6, 7, 8]
print("x3: ", x3)
print("fft x3:", fft(x3))




