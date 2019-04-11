import torch


class RoundingTransformation(object):
    """Add a gaussian noise to the tensor.

    Given tensor (C,H,W): add the Gaussian noise of the same shape.

    Source: https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv

    """

    def __init__(self, values_per_channel, round=torch.round):
        self.round_multiplier = values_per_channel - 1
        self.ext_multiplier = 1.0 / self.round_multiplier
        self.round = round

    def __call__(self, img):
        """
        Arguments:
            img (Tensor): Tensor image of size (C, H, W) to be rounded.

        Returns:
            Tensor: rounded tensor
        """
        return self.ext_multiplier * self.round(self.round_multiplier * img)
