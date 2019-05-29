"""
A simple example that shows that adding a uniform noise after the FGSM attack
can restore the correct label. The labels are given as numbers from 0 to 999.
We use ResNet-18 on 20 ImageNet samples from foolbox.

Install PyTorch 1.1: https://pytorch.org/

Install foolbox 1.9 (this version is necessary):
git clone https://github.com/bethgelab/foolbox.git
cd foolbox
git reset --hard 5191c3a595baadedf0a3659d88b48200024cd534
pip install --editable .
"""

import torch
import torchvision.models as models
import numpy as np
import foolbox

# Settings for the PyTorch model.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

imagenet_mean_array = np.array(imagenet_mean, dtype=np.float32).reshape(
    (3, 1, 1))
imagenet_std_array = np.array(imagenet_std, dtype=np.float32).reshape(
    (3, 1, 1))

# The min/max value per pixel after normalization.
imagenet_min = np.float32(-2.1179039478302)
imagenet_max = np.float32(2.640000104904175)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

# Instantiate the model.
resnet18 = models.resnet18(
    pretrained=True)
resnet18.to(device)
resnet18.eval()

model = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000,
                                    preprocessing=(imagenet_mean_array,
                                                   imagenet_std_array))
original_count = 0
adversarial_count = 0
defended_count = 0
recover_count = 20

images, labels = foolbox.utils.samples("imagenet", data_format="channels_first",
                                       batchsize=recover_count)
images = images / 255  # map from [0,255] to [0,1] range

for index, (label, image) in enumerate(zip(labels, images)):
    print("\nimage index: ", index)

    print("true prediction: ", label)

    # Original prediction of the model (without any adversarial changes or noise).
    original_predictions = model.predictions(image)
    original_prediction = np.argmax(original_predictions)
    print("original prediction: ", original_prediction)
    if original_prediction == label:
        original_count += 1

    # Attack the image.
    # attack = foolbox.attacks.FGSM(model)
    # attack = foolbox.attacks.L1BasicIterativeAttack(model)
    attack = foolbox.attacks.CarliniWagnerL2Attack(model)
    adversarial_image = attack(image, label, max_iterations=100)


    adversarial_predictions = model.predictions(adversarial_image)
    adversarial_prediciton = np.argmax(adversarial_predictions)
    print("adversarial prediction: ", adversarial_prediciton)
    if adversarial_prediciton == label:
        adversarial_count += 1

    # Add uniform noise.
    noiser = foolbox.attacks.AdditiveUniformNoiseAttack()
    noise = noiser._sample_noise(
        epsilon=0.009, image=image,
        bounds=(imagenet_min, imagenet_max))
    noised_image = adversarial_image + noise

    noise_predictions = model.predictions(noised_image)
    noise_prediction = np.argmax(noise_predictions)
    print("uniform noise prediction: ", noise_prediction)
    if noise_prediction == label:
        defended_count += 1

print(f"\nBase test accuracy of the model: "
      f"{original_count / recover_count}")
print(f"\nAccuracy of the model after attack: "
      f"{adversarial_count / recover_count}")
print(f"\nAccuracy of the model after noising: "
      f"{defended_count / recover_count}")

