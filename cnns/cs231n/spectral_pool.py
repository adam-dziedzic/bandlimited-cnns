import torch
from torch import nn

from cs231n.layers import get_out_pool_shape


def _common_spectral_pool(images, filter_size):
    print("images size: ", images.size())
    assert len(images.size()) == 5
    assert filter_size >= 3
    if filter_size % 2 == 1:
        n = int((filter_size - 1) / 2)
        top_left = images[:, :, :n + 1, :n + 1, :]
        print("top left: ", top_left)
        top_right = images[:, :, :n + 1, -n:, :]
        bottom_left = images[:, :, -n:, :n + 1, :]
        bottom_right = images[:, :, -n:, -n:, :]
        top_combined = torch.cat([top_left, top_right], dim=3)
        print("top combined size: ", top_combined.size())
        bottom_combined = torch.cat([bottom_left, bottom_right], dim=3)
        print("bottom combined size: ", bottom_combined.size())
        all_together = torch.cat([top_combined, bottom_combined], dim=2)
    else:
        all_together = images
    #     n = filter_size // 2
    #     top_left = images[:, :, :n, :n]
    #     top_middle = tf.expand_dims(
    #         tf.cast(0.5 ** 0.5, tf.complex64) *
    #         (images[:, :, :n, n] + images[:, :, :n, -n]),
    #         -1
    #     )
    #     top_right = images[:, :, :n, -(n-1):]
    #     middle_left = tf.expand_dims(
    #         tf.cast(0.5 ** 0.5, tf.complex64) *
    #         (images[:, :, n, :n] + images[:, :, -n, :n]),
    #         -2
    #     )
    #     middle_middle = tf.expand_dims(
    #         tf.expand_dims(
    #             tf.cast(0.5, tf.complex64) *
    #             (images[:, :, n, n] + images[:, :, n, -n] +
    #              images[:, :, -n, n] + images[:, :, -n, -n]),
    #             -1
    #         ),
    #         -1
    #     )
    #     middle_right = tf.expand_dims(
    #         tf.cast(0.5 ** 0.5, tf.complex64) *
    #         (images[:, :, n, -(n-1):] + images[:, :, -n, -(n-1):]),
    #         -2
    #     )
    #     bottom_left = images[:, :, -(n-1):, :n]
    #     bottom_middle = tf.expand_dims(
    #         tf.cast(0.5 ** 0.5, tf.complex64) *
    #         (images[:, :, -(n-1):, n] + images[:, :, -(n-1):, -n]),
    #         -1
    #     )
    #     bottom_right = images[:, :, -(n-1):, -(n-1):]
    #     top_combined = tf.concat(
    #         [top_left, top_middle, top_right],
    #         axis=-1
    #     )
    #     middle_combined = tf.concat(
    #         [middle_left, middle_middle, middle_right],
    #         axis=-1
    #     )
    #     bottom_combined = tf.concat(
    #         [bottom_left, bottom_middle, bottom_right],
    #         axis=-1
    #     )
    #     all_together = tf.concat(
    #         [top_combined, middle_combined, bottom_combined],
    #         axis=-2
    #     )
    return all_together


class SpectralPool(nn.Module):
    def __init__(self, filter_size=3, stride=3):
        """

        :param filter_size: the size of the filter
        :param stride: the size of the stride
        """
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, image):
        """

        :param image: numpy array representing the image
        :param filter_size: the size of the filter (one of HH, WW - there have to be the same)
        :param stride: stride for the pooling operation
        :return: image after the spectral pooling
        """
        N, C, H, W = image.size()
        out_size = get_out_pool_shape((H, W), {"pool_height": self.filter_size, "pool_width": self.filter_size,
                                               "stride": self.stride})
        onesided = False
        im_fft = torch.rfft(image, 2, onesided=onesided)
        im_transformed = _common_spectral_pool(im_fft, out_size[0])
        im_ifft = torch.irfft(im_transformed, 2, onesided=onesided)
        print("im_ifft size: ", im_ifft.size())
        # normalize image
        im_out = im_ifft
        # im_ch_last = im_ifft.permute(0, 2, 3, 1)
        # channel_max = torch.max(im_ch_last, -1)[0]
        # channel_min = torch.min(im_ch_last, -1)[0]
        # numerator = im_ch_last - channel_min
        # denominator = channel_max - channel_min
        # im_out = numerator / denominator

        return im_out
