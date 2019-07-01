import torch
import foolbox
import numpy as np
import matplotlib.pyplot as plt
from cnns.nnlib.utils.svd2d import compress_svd

# dataset = "mnist"
dataset = "imagenet"

if dataset == "imagenet":
    limx, limy = 224, 224
elif dataset == "mnist":
    limx, limy = 28, 28

images, labels = foolbox.utils.samples(dataset=dataset, index=0,
                                       batchsize=20,
                                       shape=(limx, limy),
                                       data_format='channels_first')
print("max value in images pixels: ", np.max(images))
images = images / 255
image = images[0]
format = ".png"
plt.imshow(np.moveaxis(image, 0, -1))
plt.savefig("input_image" + format)
plt.show()

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch_img = torch.tensor(image, device=device)
compress_rate = 0.1

torch_compress_img = compress_svd(torch_img=torch_img,
                                  compress_rate=compress_rate)

compress_img = torch_compress_img.cpu().numpy()

abs_diff = np.sum(np.abs(compress_img - image))
rel_diff = np.sum(np.abs(compress_img - image) / (
    np.abs(compress_img) + np.abs(image)))

print("abs diff: ", abs_diff)
print("rel diff: ", rel_diff)

plt.imshow(np.moveaxis(compress_img, 0, -1))
plt.savefig("compressed_" + str(compress_rate) + format)
plt.show()
plt.close()