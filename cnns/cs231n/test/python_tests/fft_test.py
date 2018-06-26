import numpy as np
from numpy.fft import fft, ifft

norm = None # or "orhto"

x = [8, 9, 1, 3]

print("fft(x): ", fft(x, norm=norm))

x_odd = [8, 9, 1, 3, 5]

print("fft(x_odd): ", fft(x_odd, norm=norm))

img = [[1, 2], [3, 4]]

print("fft(img): ", fft(img, norm=norm))