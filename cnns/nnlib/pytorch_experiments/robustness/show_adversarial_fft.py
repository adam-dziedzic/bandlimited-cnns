#  Band-limiting
#  Copyright (c) 2019. Adam Dziedzic
#  Licensed under The Apache License [see LICENSE for details]
#  Written by Adam Dziedzic

# if you use Jupyter notebooks
# %matplotlib inline

from cnns.nnlib.utils.general_utils import get_log_time
import matplotlib.pyplot as plt
import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
import torch
from cnns.nnlib.pytorch_layers.pytorch_utils import get_full_energy
import os
from cnns.nnlib.benchmarks.imagenet_from_class_idx_to_label import \
    imagenet_from_class_idx_to_label


def to_fft(x):
    # x = torch.from_numpy(x)
    x = torch.tensor(x)
    x = x.permute(2, 0, 1)  # move channel as the first dimension
    xfft = torch.rfft(x, onesided=False, signal_ndim=2)
    _, xfft_squared = get_full_energy(xfft)
    xfft_abs = torch.sqrt(xfft_squared)
    # xfft_abs = xfft_abs.sum(dim=0)
    return np.log(xfft_abs.numpy())


# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (np.array([104, 116, 123]), 1)
# preprocessing = (0, 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255),
                                   preprocessing=preprocessing)
# setting for the heat map
cmap = 'hot'
interpolation = 'nearest'

# get source image and label
# image, label = foolbox.utils.imagenet_example()
images, labels = foolbox.utils.samples(dataset='imagenet', index=0,
                                       batchsize=20, shape=(224, 224),
                                       data_format='channels_last')
idx = 0
image, label = images[idx], labels[idx]

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB

attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image[:, :, ::-1], label)

print("original label: id:", label, ", class: ",
      imagenet_from_class_idx_to_label[label])

# predictions_original = kmodel.predict(image[np.newaxis, :, :, ::-1])
predictions_original, _ = fmodel.predictions_and_gradient(image=image,
                                                          label=label)
original_prediction = imagenet_from_class_idx_to_label[
    np.argmax(predictions_original)]
print("model original prediction: ", original_prediction)

# predictions_adversarial = kmodel.predict(adversarial[np.newaxis, ...])
predictions_adversarial, _ = fmodel.predictions_and_gradient(
    image=adversarial, label=label)
adversarial_prediction = imagenet_from_class_idx_to_label[
    np.argmax(predictions_adversarial)]
print("model adversarial prediciton: ", adversarial_prediction)

# attack = foolbox.attacks.MultiplePixelsAttack(fmodel)
# adversarial = attack(image, label, num_pixels=100)

# attack = foolbox.attacks.AdditiveUniformNoiseAttack(fmodel)
# adversarial = attack(image[:, :, ::-1], label, epsilons=[0.4])

adversarial = adversarial[:, :, ::-1].copy()  # from BGR to RGB
# print("adversarial: ", adversarial)
# if the attack fails, adversarial will be None and a warning will be printed

if adversarial is None:
    raise Exception('foolbox did not find an adversarial example')

plt.figure()

plt.subplot(2, 3, 1)
plt.title('Original label:\n' + original_prediction)
plt.imshow(image.astype(np.uint8))
# plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Adversarial label:\n' + adversarial_prediction.replace(",", "\n"))
plt.imshow(adversarial / 255)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Difference')
difference = adversarial - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Original\nfft-ed')
original_fft = to_fft(image)
channel = 2
original_fft = original_fft[channel]
# torch.set_printoptions(profile='full')
# print("original_fft size: ", original_fft.shape)
# options = np.get_printoptions()
# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(profile='default')
# print("original_fft: ", original_fft)

# save to file
dir_path = os.path.dirname(os.path.realpath(__file__))
output_path = os.path.join(dir_path, "original_fft.csv")
print("output path: ", output_path)
np.save(output_path, original_fft)

# go back to the original print size
# np.set_printoptions(threshold=options['threshold'])
plt.imshow(original_fft, cmap=cmap, interpolation=interpolation)
heatmap_legend = plt.pcolor(original_fft)
plt.colorbar(heatmap_legend)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Adversarial\nfft-ed')
adversarial_fft = to_fft(adversarial)
adversarial_fft = adversarial_fft[channel]

output_path = os.path.join(dir_path, "adversarial_fft.csv")
print("output path: ", output_path)
np.save(output_path, adversarial_fft)

plt.imshow(adversarial_fft, cmap=cmap, interpolation=interpolation)
heatmap_legend = plt.pcolor(adversarial_fft)
plt.colorbar(heatmap_legend)
plt.axis('off')

# plt.subplot(2, 3, 6)
# plt.title('FFT Difference')
# difference_fft = adversarial_fft - original_fft
# final_difference = difference_fft / abs(difference_fft).max() * 0.2 + 0.5
# plt.imshow(final_difference)
# heatmap_legend = plt.pcolor(final_difference)
# plt.colorbar(heatmap_legend)
# plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Difference\nfft-ed')
difference = np.abs((image / 255) - (adversarial / 255))
difference = image / 255 - adversarial / 255
difference_fft = to_fft(difference)
difference_fft = difference_fft[channel]
plt.imshow(difference_fft)
heatmap_legend = plt.pcolor(difference_fft)
plt.colorbar(heatmap_legend)
plt.axis('off')

format = 'pdf'
file_name = "images/" + attack.name() + "-channel-" + str(
    channel) + "-" + get_log_time()
plt.savefig(fname=file_name + "." + format, format=format)
plt.show(block=True)
plt.close()
