from torch import nn
import torch

class SpectralPool(nn.Module):
    def __init__(self, filter_size=3, stride=3):
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, image):
        """

        :param image: numpy array representing the image
        :param filter_size: the size of the filter (one of HH, WW - there have to be the same)
        :param stride: stride for the pooling operation
        :return: image after the spectral pooling
        """
        image = torch.from_numpy(image)
        im_fft = torch.fft2d(image)
        im_transformed = _common_spectral_pool(im_fft, self.filter_size)
        im_ifft = torch.ifft2d(im_transformed)

        # normalize image
        im_ch_last =

