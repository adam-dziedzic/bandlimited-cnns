import matplotlib.pylab as plt
import numpy as np
import torch
from PIL import Image

fname = '../datasets/images/adam_spectral_pooling2.jpg'
fsave = '../datasets/images/adam_spectral_pooling_result.png'
image = Image.open(fname).convert("RGB")
init_arr = np.asarray(image)
# plt.imshow(arr, cmap='gray')
# plt.show()
print("arr initial shape: ", init_arr.shape)
# move the channel axis to the end
arr = np.moveaxis(init_arr, source=-1, destination=0)
print("arr after moveaxis shape: ", arr.shape)

# print("arr: ", arr)
print("arr type: ", arr.dtype)

arr = arr.astype(np.float32)

fig, axes = plt.subplots(3, 6, figsize=(20, 9), sharex=True, sharey=True)
pool_size = [64, 32, 16, 8, 4, 1]

for i in range(6):
    ax = axes[0, i]
    image = torch.from_numpy(arr)

    im_pool = torch.nn.functional.max_pool2d(image, kernel_size=pool_size[i])
    im_pool = im_pool.numpy()
    im_pool = im_pool.astype(np.uint8)
    CC, HH, WW = im_pool.shape
    out_image = np.zeros_like(arr).astype(np.uint8)
    print("out image shape: ", out_image.shape)
    for cc in range(CC):
        for hh in range(HH):
            for ww in range(WW):
                im_c = im_pool[cc]
                max_val = im_c[hh, ww]
                h = hh * pool_size[i]
                w = ww * pool_size[i]
                out_image[cc, h: h + pool_size[i], w: w + pool_size[i]] = max_val
    im_pool = out_image
    im_pool = np.moveaxis(im_pool, source=0, destination=-1)
    # squeeze - remove single-dimensional entries from the shape of an array
    im_pool = np.squeeze(im_pool)
    print("im_pool shape: ", im_pool.shape)
    ax.imshow(im_pool, cmap='gray')
    if not i:
        ax.set_ylabel('Max Pooling', fontsize=16)

# fig.show()
fig.savefig(fsave)
