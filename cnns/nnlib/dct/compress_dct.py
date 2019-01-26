from cnns.nnlib.pytorch_layers.test_data.cifar10_image import cifar10_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch_dct as dct
import torch


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
    """

    :param x: a tensor
    :return: a heatmap
    """
    x_numpy = x.numpy()
    print("shape of x_numpy: ", x_numpy.shape)
    data = x_numpy[0]
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()

def show_heatmap_with_gradient(x):
    x_numpy = x.numpy()
    print("shape of x_numpy: ", x_numpy.shape)
    data = x_numpy[0]
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.show()




x = cifar10_image[0]

print("CIFAR-10 image: ", x)
print("image shape: ", x.size())
# displayImage(cv2.cvtColor(x.numpy(), cv2.CAP_MODE_RGB))
# plot_random_heatmap()
# displayTensorImage(x)
show_heatmap(x)

X = dct.dct_2d(x)
print("X size:", X.size())
size = 10
X_abs = torch.abs(X)[..., :size,:size]
print("X_abs:", X_abs)
show_heatmap(X_abs)

