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

    def __init__(self, values_per_channel, mean_array, std_array,
                 device=torch.device("cpu")):
        self.values_per_channel = values_per_channel
        self.mean_array = mean_array
        self.std_array = std_array
        self.denorm = Denormalize(std_array=std_array, mean_array=mean_array, device=device)
        self.rounder = RoundingTransformation(
            values_per_channel=values_per_channel)
        self.norm = Normalize(std_array=std_array, mean_array=mean_array, device=device)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be de-normalized.

        Returns:
            Tensor: De-Normalized Tensor image.
        """
        return self.norm(self.rounder(self.denorm(tensor)))

    def round(self, numpy_array):
        """
        Execute rounding for numpy arrays. Wrapper around __call__ to call it
        for numpy arrays.

        :param numpy_array: the numpy array representing the image.
        :return: the rounded image as a numpy array.
        """
        image_attack_torch = torch.from_numpy(numpy_array)
        return self.__call__(image_attack_torch).numpy()


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, values_per_channel={2})'.format(
            self.mean_array, self.std_array, self.values_per_channel)
