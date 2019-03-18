from cnns.nnlib.pytorch_layers.test_data.cifar10_image import cifar10_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch_dct as dct
import torch
from cnns.nnlib.pytorch_layers.pytorch_utils import get_full_energy


def displayTensorImage(image):
    image = image.numpy()
    # set channels last and change to the uint8 data type
    image = np.transpose(image, (1, 2, 0)).astype("uint8")
    plt.imshow(image, interpolation="nearest")
    plt.show(block=True)


def plot_random_heatmap():
    a = np.random.random((16, 16))
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()


def show_heatmap(x, title="", file_name="heat-map"):
    """
    :param x: a tensor
    :return: a heatmap
    """
    x_numpy = x.numpy()
    print("shape of x_numpy: ", x_numpy.shape)
    # data = x_numpy[0]
    data = x_numpy.sum(axis=0)
    cmap = 'hot'
    interpolation = 'nearest'
    format = "pdf"
    plt.imshow(data, cmap=cmap, interpolation=interpolation)
    if title != "":
        plt.title(title)
    plt.show(block=True)
    plt.imsave(fname=file_name + "." + format, arr=data, cmap=cmap,
               format=format)


def show_heatmap_with_gradient(x):
    x_numpy = x.numpy()
    print("shape of x_numpy: ", x_numpy.shape)
    data = x_numpy[0]
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()


x = cifar10_image[0]

size = 32
x = x[..., :size, :size]
print("CIFAR-10 image: ", x)
print("image shape: ", x.size())

# displayImage(cv2.cvtColor(x.numpy(), cv2.CAP_MODE_RGB))
# # plot_random_heatmap()
displayTensorImage(x)
# show_heatmap(x)

xdct = dct.dct_2d(x)
print("xdct size:", xdct.size())
xdct_abs = torch.abs(xdct)
print("X_abs:", xdct_abs)
show_heatmap(xdct_abs, title="xdct_abs", file_name="xdct_abs")


xfft = torch.rfft(x, onesided=False, signal_ndim=2)
_, xfft_squared = get_full_energy(xfft)
xfft_abs = torch.sqrt(xfft_squared)
# xfft_abs = xfft_abs[..., :size, :size]
print("xfft abs: ", xfft_abs)
show_heatmap(xfft_abs, title="xfft abs", file_name="xfft_abs")  # title="xfft_abs"

