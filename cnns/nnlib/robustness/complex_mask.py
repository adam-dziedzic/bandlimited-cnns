import numpy as np
import torch


def get_val_from_interpolate(interpolate, val, ceil_init_r):
    steps = steps = int(ceil_init_r)
    if interpolate is None or interpolate == "idt" or interpolate == "const":
        get_val = lambda r, init_r: val  # apply the constant value
        steps = 0
    elif interpolate == "lin" or interpolate == "linear":
        get_val = lambda r, init_r: r / ceil_init_r
    elif interpolate == "exp" or interpolate == "exponent" or interpolate == "exponential":
        get_val = lambda r, init_r: np.exp(r - ceil_init_r)
    elif interpolate == "log" or interpolate == "logarithmic":
        get_val = lambda r, init_r: np.log(((r / ceil_init_r) * (np.e - 1)) + 1)
    else:
        raise Exception(f"Unknown interpolation: {interpolate}!")
    return get_val, steps


def get_disk_mask(side_len, compress_rate, val=0, interpolate=None):
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
    get_val, steps = get_val_from_interpolate(interpolate=interpolate,
                                              val=val,
                                              ceil_init_r=ceil_init_r)

    r = init_r
    mask = (x - ctr) ** 2 + (y - ctr) ** 2 <= r ** 2
    array_mask[mask] = 0

    # Decrease the disk size and its value of the mask in each iteration.
    for delta in range(steps):
        mask = (x - ctr) ** 2 + (y - ctr) ** 2 <= r ** 2
        array_mask[mask] = get_val(r, ceil_init_r)
        r -= 1
    # print("array mask:\n", array_mask)
    # Transform the mask to the complex representation, with 2 values for the
    # last dimension being the same.
    tensor_mask = get_tensor_mask(array_mask)
    return tensor_mask, array_mask


def get_hyper_mask(side_len, compress_rate, val=0, interpolate=None):
    # Compute the initial radius of the disk:
    # disk_area / image_area = compress_rate / 100
    # np.pi * r**2 / size_len ^^2 = compress_rate / 100
    # Hence, we compute the initial value of the radius r: init_r in the line
    # below in the following way:
    init_r = np.sqrt((1 - compress_rate / 100) * side_len ** 2 / np.pi)
    y, x = np.ogrid[0:side_len, 0:side_len]
    # Start from the slightly higher denominator for the get_val function to
    # incur some decrease in the values ( < 1.0 multiplier) for the coefficients.
    ceil_init_r = np.ceil(init_r)
    get_val, steps = get_val_from_interpolate(interpolate=interpolate,
                                              val=val,
                                              ceil_init_r=ceil_init_r)
    # Decrease the disk size and its value of the mask in each iteration.
    r = init_r
    n = side_len - 1
    # upper-left corner
    mask0 = get_array_mask(x ** 2 + y ** 2 >= r ** 2, side_len)
    # upper-right corner
    mask1 = get_array_mask((x - n) ** 2 + y ** 2 >= r ** 2, side_len)
    # bottom-left corner
    mask2 = get_array_mask(x ** 2 + (y - n) ** 2 >= r ** 2, side_len)
    # bottom-right corner
    mask3 = get_array_mask((x - n) ** 2 + (y - n) ** 2 >= r ** 2, side_len)
    array_mask = mask0 + mask1 + mask2 + mask3
    array_mask = np.clip(array_mask, a_min = 0, a_max = 1)

    for delta in range(steps):
        val = get_val(r, ceil_init_r)

        mask1 = x ** 2 + y ** 2 == r ** 2
        array_mask[mask1] = val

        mask2 = (x - (n - 1)) ** 2 + y ** 2 == r ** 2
        array_mask[mask2] = val

        mask3 = x ** 2 + (y - (n - 1)) ** 2 == r ** 2
        array_mask[mask3] = val

        mask4 = (x - (n - 1)) ** 2 + (y - (n - 1)) ** 2 == r ** 2
        array_mask[mask4] = val

        r += 1
    # print("array mask:\n", array_mask)
    # Transform the mask to the complex representation, with 2 values for the
    # last dimension being the same.
    tensor_mask = get_tensor_mask(array_mask)
    return tensor_mask, array_mask


def get_tensor_mask(array_mask):
    tensor_mask = torch.from_numpy(array_mask)
    tensor_mask = tensor_mask.unsqueeze(-1)
    tensor_mask = torch.cat((tensor_mask, tensor_mask), dim=-1)
    return tensor_mask


def get_array_mask(mask, n=7, val=0):
    array = np.ones((n, n), dtype=np.float32)
    array[mask] = val
    return array


if __name__ == "__main__":
    a, b = 1, 1
    n = 7
    ctr = n // 2  # center
    r = 2

    # y,x = np.ogrid[-a:n-a, -b:n-b]
    # y, x = np.ogrid[a:n - a, b:n - b]
    y, x = np.ogrid[0:n, 0:n]
    # print("x: ", x, " y:", y)
    mask = (x - ctr) ** 2 + (y - ctr) ** 2 <= r ** 2
    # print("mask: ", mask)

    array = np.ones((n, n))
    array[mask] = 0
    print("array: ", array)
    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(-1)
    tensor = torch.cat((tensor, tensor), dim=-1)
    # print("array: ", tensor)

    a, b = 1, 1
    n = 7
    ctr = n // 2  # center
    r = 2

    # y,x = np.ogrid[-a:n-a, -b:n-b]
    # y, x = np.ogrid[a:n - a, b:n - b]
    y, x = np.ogrid[0:n, 0:n]
    # print("x: ", x, " y:", y)
    mask1 = (x ** 2 + y ** 2 >= r ** 2)
    print("mask1: ", mask1)
    mask1 = get_array_mask(mask1)
    print("mask1: ", mask1)

    mask2 = ((x - (
            n - 1)) ** 2 + y ** 2 >= r ** 2)  # and ((x-n-1)**2 + (y-n-1)** 2 >= r ** 2) and (x**2 + (y-n-1)** 2 >= r ** 2)
    print("mask2: ", mask2)
    mask2 = get_array_mask(mask2)
    print("mask2: ", mask2)

    mask3 = ((x - (n - 1)) ** 2 + (y - (
            n - 1)) ** 2 >= r ** 2)  # and ((x-n-1)**2 + (y-n-1)** 2 >= r ** 2) and (x**2 + (y-n-1)** 2 >= r ** 2)
    print("mask3: ", mask3)
    mask3 = get_array_mask(mask3)
    print("mask3: ", mask3)

    mask4 = (x ** 2 + (y - (
            n - 1)) ** 2 >= r ** 2)  # and ((x-n-1)**2 + (y-n-1)** 2 >= r ** 2) and (x**2 + (y-n-1)** 2 >= r ** 2)
    print("mask4: ", mask4)
    mask4 = get_array_mask(mask4)
    print("mask4: ", mask4)

    array = mask1 + mask2 + mask3 + mask4

    print("array: ", array)
    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(-1)
    tensor = torch.cat((tensor, tensor), dim=-1)
