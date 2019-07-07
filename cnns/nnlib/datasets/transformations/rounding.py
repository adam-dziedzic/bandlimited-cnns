import torch


class RoundingTransformation(object):

    def __init__(self, values_per_channel, rounder=torch.round):
        """
        Compress by decreasing the number of values per channel.

        :param values_per_channel: number of values used per channel
        :param rounder: use np.around for numpy arrays
        """
        self.round_multiplier = values_per_channel - 1.0
        self.ext_multiplier = 1.0 / self.round_multiplier
        self.rounder = rounder

    def __call__(self, img):
        """
        Arguments:
            img (Tensor): Tensor image of size (C, H, W) to be rounded.

        Returns:
            Tensor: rounded tensor
        """
        return self.ext_multiplier * self.rounder(self.round_multiplier * img)
