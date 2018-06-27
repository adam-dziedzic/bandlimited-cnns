import numpy as np
from numpy.fft import fft, ifft

norm = None  # or "orhto"

x_even = np.array([8, 9, 1, 3])
print("fft(x_even): ", fft(x_even, norm=norm))
# fft(x_even):  [21.+0.j  7.-6.j -3.+0.j  7.+6.j]
fft_x_even = fft(x_even, norm=norm)
print("middle average: ", np.average([fft_x_even[1], fft_x_even[-1]]))

x_odd = np.array([8, 9, 1, 3, 5])
print("fft(x_odd): ", fft(x_odd, norm=norm))
# fft(x_odd):  [26.+0.j 9.09016994-2.62865556j -2.09016994-4.25325404j -2.09016994+4.25325404j  9.09016994+2.62865556j]

img = np.array([[1, 2], [3, 4]])
print("fft(img): ", fft(img, norm=norm))
# fft(img):  [[ 3.+0.j, -1.+0.j] [ 7.+0.j, -1.+0.j]]

img3by3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("fft(img3by3): ", fft(img3by3, norm=norm))

img4by4 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("fft(img4by4): ", fft(img4by4, norm=norm))
