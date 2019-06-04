import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

x = np.array([[6., 5., 0., 2., 4., 1.],
        [4., 2., 8., 5., 6., 8.],
        [0., 0., 4., 2., 0., 2.],
        [2., 9., 9., 9., 1., 6.],
        [3., 0., 9., 4., 6., 6.],
        [7., 3., 4., 7., 9., 0.]])
print("sum: ", x.sum())

xfft = tf.signal.rfft2d(x)
print("xfft: ", xfft.numpy())

# xfft:  tf.Tensor(
# [[153.        +0.j        -16.        -3.4641016j   0.       +10.392305j
#    11.        +0.j       ]
#  [ -4.5      +14.722432j    7.499999  +7.794228j   18.       -12.124355j
#    16.499998 +12.99038j  ]
#  [  4.499999 -19.918583j   14.       -12.124355j   16.500002  +0.866024j
#   -20.5       -0.866025j ]
#  [-45.        +0.j          9.000001  +5.196152j   -3.0000002 +1.7320508j
#     9.        -0.j       ]
#  [  4.499999 +19.918583j    3.5      -12.99038j   -11.999998 -19.05256j
#   -20.5       +0.866025j ]
#  [ -4.5      -14.722432j   11.999999 +15.588459j   -1.5      -23.382687j
#    16.499998 -12.99038j  ]], shape=(6, 4), dtype=complex64)