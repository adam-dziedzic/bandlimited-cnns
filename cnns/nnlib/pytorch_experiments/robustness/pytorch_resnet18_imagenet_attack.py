#  Band-limited CNNs
#  Copyright (c) 2019. Adam Dziedzic
#  Licensed under The Apache License [see LICENSE for details]
#  Written by Adam Dziedzic

import foolbox
import numpy as np
import torchvision.models as models

import sys

np.random.seed(31)

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

images, labels = foolbox.utils.samples(dataset='imagenet', index=0, batchsize=20,
                                       shape=(224, 224),
                                       data_format='channels_last')
images = np.transpose(images, (0, 3, 1, 2))
images = images / 255

# instantiate the model
resnet18 = models.resnet18(pretrained=True).cuda().eval()  # for CPU, remove cuda()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
model = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000,
                                    preprocessing=(mean, std))


def empty_attack(image, _):
    return image


# attack = foolbox.attacks.FGSM(model)
attack = empty_attack

print("full_attack, attack, correct (%)")
attacks = [# empty_attack,
           foolbox.attacks.SinglePixelAttack(model),
           foolbox.attacks.FGSM(model),
           foolbox.attacks.GradientAttack(model),
           foolbox.attacks.LinfinityBasicIterativeAttack(
               model, distance=foolbox.distances.MeanSquaredDistance),
           ]
for attack in attacks:
    correct = 0
    counter = 0
    for image, label in zip(images, labels):
        attack_name 
        image = attack(image, label)
        predictions = model.predictions(image)
        # print(np.argmax(predictions), label)
        if np.argmax(predictions) == label:
            correct += 1
        counter += 1
    print(str(attack), ",", , ",",
          correct / counter)
