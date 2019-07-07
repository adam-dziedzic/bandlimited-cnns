import torch

def compress_svd(torch_img, compress_rate):
    C, H, W = torch_img.size()
    assert H == W
    index = int((1 - compress_rate/100) * H)
    torch_compress_img = torch.zeros_like(torch_img)
    for c in range(C):
        try:
            u, s, v = torch.svd(torch_img[c])
        except RuntimeError as ex:
            print("SVD problem: ", ex)
            return None

        u_c = u[:, :index]
        s_c = s[:index]
        v_c = v[:, :index]

        torch_compress_img[c] = torch.mm(torch.mm(u_c, torch.diag(s_c)),
                                         v_c.t())
    return torch_compress_img


def compress_svd_numpy(numpy_array, compress_rate):
    torch_image = torch.from_numpy(numpy_array)
    torch_image = compress_svd(torch_img=torch_image,
                               compress_rate=compress_rate)
    return torch_image.cpu().numpy()


def compress_svd_batch(x, compress_rate):
    result = torch.zeros_like(x)
    for i, torch_img in enumerate(x):
        result[i] = compress_svd(torch_img=x, compress_rate=compress_rate)
    return result

