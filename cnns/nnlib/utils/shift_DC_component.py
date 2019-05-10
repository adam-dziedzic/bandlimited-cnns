import torch


def shift_DC(xfft, onesided=True, shift_to="center"):
    W = xfft.size(-2)
    H = xfft.size(-3)

    if shift_to == "center":
        HH = H // 2 + 1  # middle H
        WW = W // 2 + 1  # middle W only for bothsided (onesided = False)
    else:
        HH = H // 2
        WW = W // 2

    if onesided:
        top = xfft[..., :HH, :, :]
        bottom = xfft[..., HH:, :, :]
        return torch.cat((bottom, top), dim=-3)
    else:
        top_left = xfft[..., :HH, :WW, :]
        top_right = xfft[..., :HH, WW:, :]
        bottom_left = xfft[..., HH:, :WW, :]
        bottom_right = xfft[..., HH:, WW:, :]
        top = torch.cat((top_right, top_left), dim=-2)
        bottom = torch.cat((bottom_right, bottom_left), dim=-2)
        return torch.cat((bottom, top), dim=-3)


def shift_DC_np(xfft, onesided=True, shift_type="center"):
    xfft = torch.tensor(xfft)
    xfft = shift_DC(xfft, onesided=onesided)
    return xfft.numpy()


def shift_DC_elemwise(xfft, onesided=True):
    """
    Move the basis to the center/corner.

    :param xfft: the input tensor
    :return: the transformed tensor with moved basis.
    """
    W = xfft.size(-2)
    H = xfft.size(-3)

    out = torch.empty_like(xfft)

    HH = H // 2
    if onesided:
        WW = W
    else:
        WW = W // 2

    for row in range(H):
        for col in range(W):
            next_row = (row + HH) % H
            next_col = (col + WW) % W
            out[..., next_row, next_col, :] = xfft[..., row, col, :]

    return out
