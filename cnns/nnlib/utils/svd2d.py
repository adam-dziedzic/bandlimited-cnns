import torch
import numpy as np

from cnns import matplotlib_backend

print("Using:", matplotlib_backend.backend)
import matplotlib
import matplotlib.pyplot as plt

MY_BLUE = (56, 106, 177)
MY_RED = (204, 37, 41)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_BLACK = (83, 81, 84)

legend_position = 'upper right'
frameon = False
bbox_to_anchor = (0.0, -0.1)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


lw = 4  # the line width
ylabel_size = 25
legend_size = 20
font = {'size': 25}
title_size = 25
matplotlib.rc('font', **font)

markers = ["+", "o", "v", "s", "D", "^", "+", 'o', 'v', '+', 'v', 'D', '^', '+']
linestyles = [":", "-", "--", ":", "-", "--", "-", "--", ':', ':', "-", "-",
              "-"]


def compress_svd(torch_img, compress_rate):
    C, H, W = torch_img.size()
    assert H == W
    index = int((1 - compress_rate / 100) * H)
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
        result[i] = compress_svd(torch_img=torch_img,
                                 compress_rate=compress_rate)
    return result


iters_per_epoch = 947


def show_svd_feature_map(x, counter, args, index):
    print('counter: ', counter)
    epoch = int(counter / iters_per_epoch)
    if counter % 1 == 0:
        x = x.clone()
        x = x.detach().cpu().numpy()
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        sum_s = np.sum(s)
        print('sum of s: ', sum_s)
        with open(f'sum_singular_values_index_{index}_counter_7.txt', 'a') as f:
            f.write(f"{counter};{epoch};{sum_s}\n")
        if counter % 100 == 0:
            agg = np.mean(s, axis=0)
            agg = np.mean(agg, axis=0)
            s = show_svd_spectrum(s=agg, counter=counter, epoch=epoch,
                                  index=index)
            return s
    return None


def show_svd_spectrum(s, counter, epoch, index):
    print("singular values shape: ", s.shape)
    print("length of singular values: ", len(s))
    ax1 = plt.subplot(111)
    ax1.plot(
        range(len(s)), s, label="$\sigma_i's$",
        marker="o", linestyle="")
    ax1.set_title(f"Spectrum iteration: {counter}")
    ax1.set_xlabel("index i")
    ax1.set_ylabel("$\sigma_i$", rotation=0)
    # ax1.legend(["true rating", "predicted rating"], loc="upper left")
    # ax1.axis([0, num_train, -15, 10])
    # plt.show()
    plt.savefig(f"./svd_spectrum_{counter}_{index}.png")
    plt.close()
    return s


def show_svd_spectra(ss, counter):
    # fig = plt.figure(figsize=(15, 7))
    for i, s in enumerate(ss):
        with open(f'mnist_singular_values_{i}.txt', 'ab') as f:
            np.savetxt(f, s[np.newaxis, :], delimiter=';')
    ax1 = plt.subplot(111)
    for i, s in enumerate(ss):
        ax1.plot(
            range(len(s)),
            s,
            label=f"conv{i + 1}",
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)])
    ax1.set_title(f"Iteration: {counter}")
    ax1.set_xlabel("index i")
    ax1.set_ylabel("$\sigma_i$", rotation=0)
    # ax1.legend(["true rating", "predicted rating"], loc="upper left")
    # ax1.axis([0, num_train, -15, 10])
    # plt.show()
    plt.legend(loc=legend_position,
               frameon=frameon,
               prop={'size': legend_size},
               # bbox_to_anchor=bbox_to_anchor,
               ncol=1,
               )
    plt.tight_layout()
    plt.savefig(f"./svd_spectra_{counter}.png")
    plt.close()
    return ss
