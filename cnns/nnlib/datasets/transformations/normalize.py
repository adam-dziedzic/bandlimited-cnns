import torch


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) * std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, device=torch.device('cpu')):
        self.mean = torch.tensor(mean, device=device).view(3, 1, 1)
        self.std = torch.tensor(std, device=device).view(3, 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be de-normalized.

        Returns:
            Tensor: De-Normalized Tensor image.
        """
        # return tensor.sub_(self.mean).div_(self.std)
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean,
                                                                      self.std)
