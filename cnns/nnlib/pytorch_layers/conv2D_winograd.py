import sys
import torch
from torch import tensor


class Winograd(object):

    def __init__(self, filter_value=None):
        super(Winograd, self).__init__()

        if filter_value is not None:
            self.filter = filter_value
        self.B = tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, -1.0, 1.0],
             [-1.0, 1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, -1.0]])
        self.B_T = self.B.transpose(1, 0)
        self.G = tensor(
            [[1.0, 0.0, 0.0],
             [0.5, 0.5, 0.5],
             [0.5, -0.5, 0.5],
             [0.0, 0.0, 1.0]])
        self.G_T = self.G.transpose(1, 0)
        self.A = tensor([[1.0, 0.0],
                         [1.0, 1.0],
                         [1.0, -1.0],
                         [0.0, -1.0]])
        self.A_T = self.A.transpose(1, 0)


    def winograd_F_2_3(self, input, filter):
        """
        Compute winograd convolution with output of size 2x2 and filter of size
        3x3.

        :param input: 4x4
        :param filter: 3x3
        :return: 2x2
        """
        U = torch.matmul(self.G, torch.matmul(filter, self.G_T))
        V = torch.matmul(self.B_T, torch.matmul(input, self.B))
        return torch.matmul(self.A_T, torch.matmul(U*V, self.A))

if __name__ == "__main__":
    import doctest

    sys.exit(doctest.testmod()[0])
