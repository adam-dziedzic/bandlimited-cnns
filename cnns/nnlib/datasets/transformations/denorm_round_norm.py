import torch
from cnns.nnlib.datasets.transformations.denormalize import Denormalize
from cnns.nnlib.datasets.transformations.normalize import Normalize
from cnns.nnlib.datasets.transformations.rounding import RoundingTransformation


class DenormRoundNorm(object):
    """De-Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will denormalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = input[channel] * std[channel] + mean[channel]``
    Then round and normalize it.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, values_per_channel, mean, std, device):
        self.values_per_channel = values_per_channel
        self.mean = mean
        self.std = std
        self.denorm = Denormalize(std=std, mean=mean, device=device)
        self.round = RoundingTransformation(
            values_per_channel=values_per_channel)
        self.norm = Normalize(std=std, mean=mean, device=device)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be de-normalized.

        Returns:
            Tensor: De-Normalized Tensor image.
        """
        return self.norm(self.round(self.denorm(tensor)))

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, values_per_channel={2})'.format(
            self.mean, self.std, self.values_per_channel)
