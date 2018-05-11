import numpy as np
from scipy import signal

# source: https://stackoverflow.com/questions/40703751/using-fourier-transforms-to-do-convolution?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

x = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]]
x = np.array(x)
y = [[4, 5], [3, 4]]
y = np.array(y)

standard_conv = signal.convolve2d(x, y, 'full')

print("conv:", standard_conv)

s1 = np.array(x.shape)
s2 = np.array(y.shape)
print("s1: ", s1)
print("s2: ", s2)

size = s1 + s2 - 1
print("size: ", size)

fsize = 2 ** np.ceil(np.log2(size)).astype(int)
fslice = tuple([slice(0, int(sz)) for sz in size])
print("fslice: ", fslice)

# Along each axis, if the given shape (fsize) is smaller than that of the input, the input is cropped.
# If it is larger, the input is padded with zeros. if s is not given, the shape of the input along the axes
# specified by axes is used.
new_x = np.fft.fft2(x, fsize)

new_y = np.fft.fft2(y, fsize)
result = np.fft.ifft2(new_x * new_y)
print("first result: ", result)

result = np.fft.ifft2(new_x * new_y)[fslice].copy()
result_int = np.array(result.real, np.int32)

my_result = np.array(result, np.double)
print("my_result (doubles): ", my_result)

print("fft for my method (ints):", result_int)
print("is my method correct (for ints): ", np.array_equal(result_int, standard_conv))
print("fft for my method (doubles):", result)

print("fft with int32 output:", np.array(signal.fftconvolve(x, y), np.int32))
lib_result = np.array(signal.fftconvolve(x, y), np.double)
print("fft with double output:", np.allclose(my_result, lib_result, atol=1e-12))

# the correct way is to take the amplitude:  the abs of a complex number gives us its amplitude/mangnitude
lib_magnitude = np.abs(signal.fftconvolve(x, y))
print("lib_magnitude: ", lib_magnitude)
my_magnitude = np.abs(result)
print("is the magnitude correct: ", np.allclose(my_magnitude, lib_magnitude, atol=1e-12))
