# if you use Jupyter notebooks
# %matplotlib inline

import matplotlib.pyplot as plt
import foolbox
import numpy as np
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_numpy_through_torch
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_numpy
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_resize
from cnns.nnlib.robustness.channels.channels_definition import fft_numpy


# instantiate model
preprocessing = (np.array([104, 116, 123]), 1)

# get source image and label
image, label = foolbox.utils.imagenet_example()
image = image / 255  # # division by 255 to convert [0, 255] to [0, 1]
print('image dimensions: ', image.shape)
compress_rate = 90
image_CHW = np.moveaxis(image, -1, 0)
compress_type = 'fft'  # 'torch' 'resize'

if compress_type == "svd_torch":
    image_CHW = compress_svd_numpy_through_torch(numpy_array=image_CHW,
                                                 compress_rate=compress_rate)
elif compress_type == "svd_numpy":
    image_CHW = compress_svd_numpy(numpy_array=image_CHW,
                                   compress_rate=compress_rate)
elif compress_type == "svd_resize":
    image_CHW = compress_svd_resize(numpy_array=image_CHW,
                                    compress_rate=compress_rate)
elif compress_type == "fft":
    image_CHW = fft_numpy(numpy_array=image_CHW,
                          compress_rate=compress_rate)

image_compressed = np.moveaxis(image_CHW, 0, -1)
image_compressed = np.clip(image_compressed, a_min=0.0, a_max=1.0)
plt.figure()

plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'{compress_type} compressed {compress_rate}%')
plt.imshow(image_compressed)
plt.axis('off')

plt.show()
