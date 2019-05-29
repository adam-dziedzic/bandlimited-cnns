import torchvision.models as models
import numpy as np
import foolbox
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_min
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_max
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_mean_array
from cnns.nnlib.datasets.imagenet.imagenet_pytorch import imagenet_std_array

# instantiate the model
resnet18 = models.resnet18(
    pretrained=True).cuda().eval()  # for CPU, remove cuda()

model = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000,
                                    preprocessing=(imagenet_mean_array,
                                                   imagenet_std_array))

for index in range(20):
    print("\n image index: ", index)
    image, label = foolbox.utils.samples("imagenet", index=index,
                                         data_format="channels_first")
    image = image / 255  # map from [0,255] to [0,1]

    # no batch dimension
    image = image[0]
    label = label[0]

    print("true prediction: ", label)

    # Original prediction of the model (without any adversarial changes or noise).
    original_predictions = model.predictions(image)
    print("original prediction: ", np.argmax(original_predictions))

    # Attack the image.
    attack = foolbox.attacks.FGSM(model)
    adversarial_image = attack(image, label)

    adversarial_predictions = model.predictions(adversarial_image)
    print("adversarial prediction: ", np.argmax(adversarial_predictions))


    # Add uniform noise.
    noiser = foolbox.attacks.AdditiveUniformNoiseAttack()
    noise = noiser._sample_noise(
                    epsilon=0.009, image=image,
                    bounds=(imagenet_min, imagenet_max))
    randomized_image = adversarial_image + noise

    noise_predictions = model.predictions(randomized_image)
    print("uniform noise prediction: ", np.argmax(noise_predictions))


images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=20,
                                     shape=(224, 224),
                                     data_format='channels_last')
print(image.shape)
images = np.transpose(images, (0, 3, 1, 2))
print(image.shape)
images = images / 255

predictions = model.batch_predictions(images)
print("predictions: ", predictions.shape)
print(np.argmax(predictions, axis=1), labels)
