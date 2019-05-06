import numpy as np
import torch


def get_complex_mask(side_len, compress_rate, val=0, interpolate=None):
    # Compute the initial radius of the disk:
    # disk_area / image_area = compress_rate / 100
    # np.pi * r**2 / size_len ^^2 = compress_rate / 100
    # Hence, we compute the initial value of the radius r: init_r in the line
    # below in the following way:
    init_r = np.sqrt((compress_rate / 100) * side_len ** 2 / np.pi)
    y, x = np.ogrid[0:side_len, 0:side_len]
    ctr = side_len // 2  # center
    array_mask = np.ones((side_len, side_len), dtype=np.float32)
    # Start from the slightly higher denominator for the get_val function to
    # incur some decrease in the values ( < 1.0 multiplier) for the coefficients.
    ceil_init_r = np.ceil(init_r)
    steps = int(ceil_init_r)
    if interpolate is None or interpolate == "idt" or interpolate == "const":
        steps = 1
        get_val = lambda r, init_r: val  # apply the constant value
    elif interpolate == "lin" or interpolate == "linear":
        get_val = lambda r, init_r: r / ceil_init_r
    elif interpolate == "exp" or interpolate == "exponent" or interpolate == "exponential":
        get_val = lambda r, init_r: np.exp(r - ceil_init_r)
    elif interpolate == "log" or interpolate == "logarithmic":
        get_val = lambda r, init_r: np.log(((r / ceil_init_r) * (np.e - 1)) + 1)
    else:
        raise Exception(f"Unknown interpolation: {interpolate}!")
    # Decrease the disk size and its value of the mask in each iteration.
    r = init_r
    for delta in range(steps):
        mask = (x - ctr) ** 2 + (y - ctr) ** 2 <= r ** 2
        array_mask[mask] = get_val(r, ceil_init_r)
        r -= 1
    # print("array mask:\n", array_mask)
    # Transform the mask to the complex representation, with 2 values for the
    # last dimension being the same.
    tensor_mask = torch.from_numpy(array_mask)
    tensor_mask = tensor_mask.unsqueeze(-1)
    tensor_mask = torch.cat((tensor_mask, tensor_mask), dim=-1)
    return tensor_mask, array_mask


if __name__ == "__main__":
    a, b = 1, 1
    n = 7
    ctr = n // 2  # center
    r = 2

    # y,x = np.ogrid[-a:n-a, -b:n-b]
    # y, x = np.ogrid[a:n - a, b:n - b]
    y, x = np.ogrid[0:n, 0:n]
    print("x: ", x, " y:", y)
    mask = (x - ctr) ** 2 + (y - ctr) ** 2 <= r ** 2
    print("mask: ", mask)

    array = np.ones((n, n))
    array[mask] = 0
    print("array: ", array)
    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(-1)
    tensor = torch.cat((tensor, tensor), dim=-1)
    print("array: ", tensor)
