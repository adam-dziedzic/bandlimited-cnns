import torch
from torchvision.transforms import Normalize
from cnns.nnlib.datasets.transformations.denormalize import Denormalize
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

    def __init__(self, mean, std, values_per_channel):
        self.values_per_channel = values_per_channel
        self.denorm = Denormalize(std=torch.tensor(std).view(3,1,1),
                                  mean=torch.tensor(mean).view(3,1,1))
        self.round = RoundingTransformation(values_per_channel=values_per_channel)
        self.norm = Normalize(std=std, mean=mean)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be de-normalized.

        Returns:
            Tensor: De-Normalized Tensor image.
        """
        return self.norm(self.round(self.denorm(tensor)))

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean,
                                                                      self.std)
