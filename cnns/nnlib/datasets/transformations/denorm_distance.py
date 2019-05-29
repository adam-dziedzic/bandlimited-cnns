import torch
from cnns.nnlib.datasets.transformations.denormalize import Denormalize


class DenormDistance(object):
    """De-Normalize 2 tensors with mean and standard deviation. Go to the [0,1]
    range of values. Calculate distance between them

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean_array, std_array, distance_metric=torch.norm,
                 device=torch.device("cpu")):
        self.distance_metric = distance_metric
        self.mean_array = mean_array
        self.std_array = std_array
        self.denorm = Denormalize(std_array=std_array, mean_array=mean_array,
                                  device=device)

    def __call__(self, tensor1, tensor2, norm=2, dim=None):
        """
        Args:
            tensor1 (Tensor): Tensor image of size (C, H, W) to be de-normalized.
            tensor2 (Tensor): Tensor image of size (C, H, W) to be de-normalized.
        Returns:
            Tensor: distance between images
        """
        return self.distance_metric(
            (self.denorm(tensor1) - self.denorm(tensor2)),
            p=norm, dim=dim).item()

    def measure(self, numpy_array1, numpy_array2, norm=2, dim=None):
        """
        Calculate distance measure for numpy arrays. Wrapper around __call__ to
        call it for numpy arrays.

        :param numpy_array1: the numpy array representing the image.
        :param numpy_array2: the numpy array representing the image.
        :return: the distance between the images
        """
        return self.__call__(torch.from_numpy(numpy_array1),
                             torch.from_numpy(numpy_array2), norm, dim=dim)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, ' \
                                         'distance_metric={2})'.format(
            self.mean_array, self.std_array, self.distance_metric)


if __name__ == "__main__":
    t1 = torch.tensor([[1.0,2,3]])
    t2_1 = [[2.0,2,3]]
    t2_2 = [[1.0, 3,3]]

    t2_1t = torch.tensor(t2_1)
    print("dist1: ", torch.dist(t1, t2_1t))

    t2_2t = torch.tensor(t2_2)
    print("dist1: ", torch.dist(t1, t2_2t))

    t2 = torch.tensor([t2_1, t2_2])
    dist_all = torch.dist(t1, t2).item()
    print("dist all: ", dist_all)
    print(t1-t2)

    dist_elem_wise = torch.norm(t1 - t2, dim=0)
    print("dist_elem_wise: ", dist_elem_wise)
