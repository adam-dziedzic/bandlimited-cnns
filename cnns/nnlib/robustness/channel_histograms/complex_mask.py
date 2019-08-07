import numpy as np
import torch


def get_val_from_interpolate_disk(interpolate, val, ceil_init_r):
    steps = int(ceil_init_r)
    if interpolate is None or interpolate == "idt" or interpolate == "const":
        get_val = lambda r, init_r: val  # apply the constant value
        steps = 0
    elif interpolate == "lin" or interpolate == "linear":
        get_val = lambda r: r / ceil_init_r
    elif interpolate == "exp" or interpolate == "exponent" or interpolate == "exponential":
        get_val = lambda r: np.exp(r - ceil_init_r)
    elif interpolate == "log" or interpolate == "logarithmic":
        get_val = lambda r: np.log(((r / ceil_init_r) * (np.e - 1)) + 1)
    else:
        raise Exception(f"Unknown interpolation: {interpolate}!")
    return get_val, steps


def get_val_from_interpolate_hyper(interpolate, val, start_r, stop_r):
    """
    :param interpolate: the type of value interpolation
    :param val: the value to be set
    :param stop_r: the starting value of the radius
    :param start_r:  the stoping value of the radius
    :return: the value interpolation function, number of total steps
    """
    steps = (stop_r - start_r) + 1
    delta_r = (stop_r - start_r) + 1
    if interpolate is None or interpolate == "idt" or interpolate == "const":
        get_val = lambda r: val  # apply the constant value
        steps = 0
    elif interpolate == "lin" or interpolate == "linear":
        get_val = lambda r: (stop_r - r) / delta_r
    elif interpolate == "exp" or interpolate == "exponent" or interpolate == "exponential":
        get_val = lambda r: np.exp(start_r - r - 1)
    elif interpolate == "log" or interpolate == "logarithmic":
        get_val = lambda r: np.log(
            (((stop_r - r) / delta_r) * (np.e - 1)) + 1)
    else:
        raise Exception(f"Unknown interpolation: {interpolate}!")
    return get_val, steps


def get_disk_mask(H, W, compress_rate, val=0, interpolate=None, onesided=True):
    # Compute the initial radius of the disk:
    # disk_area / image_area = compress_rate / 100
    # np.pi * r**2 / size_len ^^2 = compress_rate / 100
    # Hence, we compute the initial value of the radius r: init_r in the line
    # below in the following way:
    side_len = H
    # There is only a slight difference in the areas between the onesided and
    # non-onesided cases but we perform the exact computation to limit the
    # errors to minimum.
    # if onesided:
    #     init_r = np.sqrt(2 * (compress_rate / 100) * H * W / np.pi)
    if onesided is False:
        if H != W:
            raise Exception("We only support squared inputs.")

    init_r = np.sqrt((compress_rate / 100) * side_len ** 2 / np.pi)

    y, x = np.ogrid[0:side_len, 0:side_len]
    ctr = side_len // 2  # center
    array_mask = np.ones((side_len, side_len), dtype=np.float32)
    # Start from the slightly higher denominator for the get_val function to
    # incur some decrease in the values ( < 1.0 multiplier) for the coefficients.
    ceil_init_r = np.ceil(init_r)
    get_val, steps = get_val_from_interpolate_disk(interpolate=interpolate,
                                                   val=val,
                                                   ceil_init_r=ceil_init_r)

    r = init_r
    mask = (x - ctr) ** 2 + (y - ctr) ** 2 <= r ** 2
    array_mask[mask] = 0

    # Decrease the disk size and its value of the mask in each iteration.
    for delta in range(steps):
        mask = (x - ctr) ** 2 + (y - ctr) ** 2 <= r ** 2
        array_mask[mask] = get_val(r)
        r -= 1
    # print("array mask:\n", array_mask)
    # Transform the mask to the complex representation, with 2 values for the
    # last dimension being the same.
    tensor_mask = get_tensor_mask(array_mask)
    return tensor_mask, array_mask


def get_inverse_hyper_mask(H, W, compress_rate, val=0, interpolate=None,
                           onesided=True):
    mask, array_mask = get_hyper_mask(
        H=H, W=W, compress_rate=compress_rate, val=val,
        interpolate=interpolate, onesided=onesided)
    mask = mask * (-1) + 1
    array_mask = array_mask * (-1) + 1
    return mask, array_mask


def get_hyper_mask(H, W, compress_rate, val=0, interpolate=None,
                   onesided=True):
    # Compute the initial radius of the hyperdisk for onesided=False, the whole
    # FFT map is returned (not-halved).
    # hyper_area / image_area = (1 - compress_rate / 100)
    # np.pi * r**2 / size_len ^^2 = (1 - compress_rate / 100)
    # Hence, we compute the initial value of the radius r: init_r in the line
    # below in the following way:
    side_len = H
    # There is only a slight difference in the areas between the onesided and

    # non-onesided cases but we perform the exact computation to limit the
    # errors to minimum.
    # if onesided:
    #     init_r = np.sqrt(2 * (1 - compress_rate / 100) * H * W / np.pi)
    if onesided is False:
        if H != W:
            raise Exception("We only support squared inputs.")

    if onesided == True:
        multiplier = 2
        WW = W
    else:
        multiplier = 1
        WW = W / 2

    compress_rate = compress_rate / 100

    start_r = np.sqrt(multiplier * (1 - compress_rate) * H * W / np.pi)
    stop_r = np.sqrt((H / 2) ** 2 + WW ** 2)

    start_r = np.floor(start_r)
    stop_r = np.ceil(stop_r)

    y, x = np.ogrid[0:side_len, 0:side_len]
    get_val, steps = get_val_from_interpolate_hyper(interpolate=interpolate,
                                                    val=val,
                                                    start_r=start_r,
                                                    stop_r=stop_r)
    # Decrease the disk size and its value of the mask in each iteration.
    r = start_r
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
    # If the compression is small, the ones can appear in the same positions for
    # different masks, so limit them to just 1 and does not allow to exceed the
    # bound (otherwise, two-s can emerge).
    array_mask = np.clip(array_mask, a_min=0, a_max=1)

    if steps > 0:
        # Prepare the quadrant masks, so that the mask(x) below can only change values
        # in its quadrant.
        if H % 2 == 0:
            mid = H // 2
        else:
            mid = H // 2 + 1

        # upper-left corner
        quadrant0 = get_array_mask(np.outer(x < mid, y < mid), side_len, val=1,
                                   back=0)
        # upper-right corner
        quadrant1 = get_array_mask(np.outer(x < mid, y >= mid), side_len, val=1,
                                   back=0)
        # bottom-left corner
        quadrant2 = get_array_mask(np.outer(x >= mid, y < mid), side_len, val=1,
                                   back=0)
        # bottom-right corner
        quadrant3 = get_array_mask(np.outer(x >= mid, y >= mid), side_len,
                                   val=1, back=0)

    mask0 = np.ones((side_len, side_len))
    mask1 = np.ones((side_len, side_len))
    mask2 = np.ones((side_len, side_len))
    mask3 = np.ones((side_len, side_len))

    r = start_r
    for delta in range(int(steps)):
        val = get_val(r)

        mask0 = get_array_mask(
            mask=(x ** 2 + y ** 2 >= r ** 2),
            n=side_len,
            val=val,
            back=None,
            array_mask=mask0)

        mask1 = get_array_mask(
            mask=((x - n) ** 2 + y ** 2 >= r ** 2),
            n=side_len,
            val=val,
            back=None,
            array_mask=mask1)

        mask2 = get_array_mask(
            mask=(x ** 2 + (y - n) ** 2 >= r ** 2),
            n=side_len,
            val=val,
            back=None,
            array_mask=mask2)

        mask3 = get_array_mask(
            mask=((x - n) ** 2 + (y - n) ** 2 >= r ** 2),
            n=side_len,
            val=val,
            back=None,
            array_mask=mask3)

        r += 1

    if steps > 0:
        mask0 *= quadrant0
        mask1 *= quadrant1
        mask2 *= quadrant2
        mask3 *= quadrant3

        array_mask = mask0 + mask1 + mask2 + mask3

    # print("array mask:\n", array_mask)
    # Transform the mask to the complex representation, with 2 values for the
    # last dimension being the same.
    tensor_mask = get_tensor_mask(array_mask)
    return tensor_mask, array_mask


def get_tensor_mask(array_mask):
    tensor_mask = torch.from_numpy(array_mask)
    tensor_mask = tensor_mask.unsqueeze(-1)
    # tensor_other = torch.zeros_like(tensor_mask)
    tensor_other = tensor_mask
    tensor_mask = torch.cat((tensor_mask, tensor_other), dim=-1)
    return tensor_mask


def get_array_mask(mask, n=7, val=0, back=1, dtype=np.float32,
                   array_mask=None):
    if back is None:
        if array_mask is None:
            raise Exception("array_mask has to be specified if back is None")
        array = array_mask
    elif back == 1:
        array = np.ones((n, n), dtype=dtype)
    elif back == 0:
        array = np.zeros((n, n), dtype=dtype)
    else:
        array = np.full((n, n), fill_value=back, dtyp=dtype)
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
