import sys
import numpy
import foolbox
from numpy.linalg import svd
numpy.set_printoptions(threshold=sys.maxsize)

image, label = foolbox.utils.samples(
    dataset='cifar10', index=0, batchsize=1, data_format='channels_first')
image = image / 255  # # division by 255 to convert [0, 255] to [0, 1]

u, s, vh = svd(a=image, full_matrices=False)
print('label: ', label)
print('u: ', u)
print('s: ', s)
print('vh: ', vh)
