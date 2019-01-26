from cnns.nnlib.pytorch_layers.test_data.cifar10_image import cifar10_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch_dct as dct


def displayTensorImage(image):
    image = image.numpy()
    # set channels last and change to the uint8 data type
    image = np.transpose(image, (1, 2, 0)).astype("uint8")
    plt.imshow(image.T, interpolation="nearest")
    plt.show()


def plot_random_heatmap():
    a = np.random.random((16, 16))
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()


def show_heatmap(x):
    x_numpy = x.numpy()
    print("shape of x_numpy: ", x_numpy.shape)
    plt.imshow(x_numpy[0], cmap='hot', interpolation='nearest')
    plt.show()


x = cifar10_image[0]

print("CIFAR-10 image: ", x)
print("image shape: ", x.size())
# displayImage(cv2.cvtColor(x.numpy(), cv2.CAP_MODE_RGB))
# plot_random_heatmap()
# displayTensorImage(x)
# show_heatmap(x)

X = dct.dct_2d(x)
show_heatmap(X)
