from torch.distributions.normal import Normal
import torch


def gauss(image_numpy, sigma):
    """
    Add Gaussian noise with strength sigma to the input image_numpy.

    :param image_numpy: the input image as a numpy array.
    :param sigma: the level of Gaussian noise.
    :return: the noised image.
    """
    transformer = AddGaussianNoiseTransformation(sigma)
    return transformer.gauss(image_numpy)


class AddGaussianNoiseTransformation(object):
    """Add a gaussian noise to the tensor.

    Given tensor (C,H,W): add the Gaussian noise of the same shape.

    Source: https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv

    """

    def __init__(self, sigma=0):
        self.sigma = sigma
        self.m = Normal(0, sigma)

    def __call__(self, data_item):
        """
        Arguments:
            data_item (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: tensor with added gaussian noise
        """
        # print("data_item device: ", data_item.device, data_item.dtype, data_item.size())
        # print("data_item: ", data_item)
        data_item += self.m.sample(data_item.size())
        return data_item

    def gauss(self, numpy_array):
        """
        Wrapper around __call__ to call it for numpy arrays.

        :param numpy_array: the numpy array representing the image.
        :return: the image as a numpy array.
        """
        image_torch = torch.from_numpy(numpy_array)
        return self.__call__(image_torch).numpy()
