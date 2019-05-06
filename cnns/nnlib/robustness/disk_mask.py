import numpy as np
import torch


def get_complex_mask(side_len, compress_rate, val=0):
    r = np.sqrt((compress_rate / 100) * side_len ** 2 / np.pi)
    y, x = np.ogrid[0:side_len, 0:side_len]
    ctr = side_len // 2  # center
    mask = (x - ctr) ** 2 + (y - ctr) ** 2 <= r ** 2
    array = np.ones((side_len, side_len), dtype=np.float32)
    array[mask] = val
    # print("array: ", array)
    # Transform the mask to the complex representation, with 2 values for the
    # last dimension being the same.
    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(-1)
    tensor = torch.cat((tensor, tensor), dim=-1)
    return tensor


if __name__ == "__main__":
    a, b = 1, 1
    n = 7
    ctr = n // 2  # center
    r = 2

    # y,x = np.ogrid[-a:n-a, -b:n-b]
    # y, x = np.ogrid[a:n - a, b:n - b]
    y, x = np.ogrid[0:n, 0:n]
    print("x: ", x, " y:", y)
    mask = (x-ctr)**2 + (y-ctr)**2 <= r**2
    print("mask: ", mask)

    array = np.ones((n, n))
    array[mask] = 0
    print("array: ", array)
    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(-1)
    tensor = torch.cat((tensor, tensor), dim=-1)
    print("array: ", tensor)
