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

    def __init__(self, mean_array, std_array, device=torch.device('cpu')):
        self.mean_array = torch.tensor(mean_array, device=device)
        self.std_array = torch.tensor(std_array, device=device)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be de-normalized.

        Returns:
            Tensor: De-Normalized Tensor image.
        """
        # return tensor.sub_(self.mean).div_(self.std)
        return (tensor - self.mean_array) / self.std_array

    def normalize(self, numpy_array):
        """
        Wrapper around __call__ to call it for numpy arrays.

        :param numpy_array: the numpy array representing the image.
        :return: the image as a numpy array.
        """
        image_torch = torch.from_numpy(numpy_array)
        return self.__call__(image_torch).numpy()

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean_array,
            self.std_array)
