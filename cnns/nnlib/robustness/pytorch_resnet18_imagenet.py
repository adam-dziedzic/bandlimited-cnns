#  Band-limited CNNs
#  Copyright (c) 2019. Adam Dziedzic
#  Licensed under The Apache License [see LICENSE for details]
#  Written by Adam Dziedzic

import torchvision.models as models
import numpy as np
import foolbox

# instantiate the model
resnet18 = models.resnet18(
    pretrained=True).cuda().eval()  # for CPU, remove cuda()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
model = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000,
                                    preprocessing=(mean, std))

image, label = foolbox.utils.imagenet_example(data_format='channels_first')
print(image.shape)
image = image / 255

attack = foolbox.attacks.FGSM(model)
image = attack(image, label)

print("single prediction:", np.argmax(model.predictions(image)))
print("correct label: ", label)


images, labels = foolbox.utils.samples(dataset='imagenet', index=0, batchsize=20,
                                     shape=(224, 224),
                                     data_format='channels_last')
print(image.shape)
images = np.transpose(images, (0, 3, 1, 2))
print(image.shape)
images = images / 255

predictions = model.batch_predictions(images)
print("predictions: ", predictions.shape)
print(np.argmax(predictions, axis=1), labels)
