from torch.distributions.normal import Normal

class AddGaussianNoiseTransformation(object):
    """Add a gaussian noise to the tensor.

    Given tensor (C,H,W): add the Gaussian noise of the same shape.

    Source: https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv

    """
    def __init__(self, sigma=0):
        self.sigma = sigma
        self.m = Normal(0, sigma)

    def __call__(self, data_item):
        """
        Arguments:
            data_item (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: tensor with added gaussian noise
        """
        # print("data_item device: ", data_item.device, data_item.dtype, data_item.size())
        # print("data_item: ", data_item)
        data_item += self.m.sample(data_item.size())
        return data_item