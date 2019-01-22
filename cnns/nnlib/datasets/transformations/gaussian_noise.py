from torch.distributions.normal import Normal

class AddGaussianNoiseTransformation(object):
    """Add a gaussian noise to the tensor.

    Given tensor (C,H,W): add the Gaussian noise of the same shape.

    Source: https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv

    """
    def __init__(self, sigma=0):
        self.sigma = sigma
        self.m = Normal(0, sigma)

    def __call__(self, tensor):
        """
        Arguments:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: tensor with added gaussian noise
        """
        tensor += self.m.sample(tensor.size())
        return tensor