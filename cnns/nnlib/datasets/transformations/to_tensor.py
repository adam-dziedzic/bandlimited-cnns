import torch


class ToTensorWithType(object):

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, ndarray):
        """
        Arguments:
            tensor (ndarray): numpy array.

        Returns:
            Tensor: with dtype.
        """
        tensor = torch.from_numpy(ndarray)
        tensor = tensor.to(dtype=self.dtype)
        return tensor
