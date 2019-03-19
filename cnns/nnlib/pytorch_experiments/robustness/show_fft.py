import numpy as np
import matplotlib.pyplot as plt
import os

# setting for the heat map
cmap = 'hot'
interpolation = 'nearest'

dir_path = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(dir_path, "original_fft.csv.npy")
original_fft = np.load(output_path)

fft_img = original_fft
min = fft_img.min()
max = fft_img.max()

# normalize the image
# fft_img = (fft_img - min) / (max - min)

# fft_img = np.random.random((16, 16))

limit_size = 8
fft_img = fft_img[:limit_size,:limit_size]
fft_img = np.log(fft_img)
# print(fft_img)

plt.imshow(fft_img, cmap=cmap, interpolation=interpolation)
heatmap_legend = plt.pcolor(fft_img)
plt.colorbar(heatmap_legend)
plt.show(block=True)

output_path = os.path.join(dir_path, "adversarial_fft.csv.npy")
adversarial_fft = np.load(output_path)
fft_img = adversarial_fft
fft_img = fft_img[:limit_size,:limit_size]
fft_img = np.log(fft_img)
plt.imshow(fft_img, cmap=cmap, interpolation=interpolation)
heatmap_legend = plt.pcolor(fft_img)
plt.colorbar(heatmap_legend)
plt.show(block=True)