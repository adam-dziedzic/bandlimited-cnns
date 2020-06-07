# Analysis of Perturbation Based Defenses

The existence of adversarial examples, or intentional mis-predictions constructed from small changes to correctly predicted examples, is one of the most significant challenges in neural network research today.
Ironically, many new defenses are based on a simple observation---the adversarial inputs themselves are not robust and small perturbations to the attacking input often recover the desired prediction.
While the intuition is somewhat clear, a detailed understanding of this phenomenon is missing from the research literature.
This paper presents a comprehensive experimental analysis of when and why perturbation defenses work and potential mechanisms that could explain their effectiveness (or ineffectiveness) in different settings.

## Code navigation

The main file of the project is in cnns/nnlib/robustness/main_adversarial.py.

The compression in the frequency domain is implemented in: cnns/nnlib/pytorch_layers/fft_band_2D_complex_mask.py You can also find: noise.py and round.py layers in the same folder for the noisy channel and feature squeezing (color depth reduction, also known as rounding).

The arguments are listed in cnns/nnlib/utils/arguments.py and cnns/nnlib/utils/exec_args.py.

The additional compression layers CD, FC and stochastic channel are in: stochastic-channels/cnns/nnlib/pytorch_layers with names: fft_band_2D_complex_mask.py, round.py, and noise.py, respectively.

We had to implement a separate adaptive adversarial attack in stochastic-channels/cnns/nnlib/attacks/carlini_wagner_round_fft.py because the many trials for the noisy channel require many inference operations through the network, so this cannot be implemented simply as a layer of a neural network. 

The code that compares RobustNet with PNI and adversarial training is in folder: cnns/nnlib/robustness/pni/code

The EOT for PGD is implemented in: cnns/nnlib/robustness/pni/code/models/attack_model.py

```
    def pgd(model, data, target, k=7, a=0.01, random_start=True,
            d_min=0, eot=1, d_max=1, eot=None, epsilon=8/255):
        """

        :param model: the model to attack
        :param data: clean input data
        :param target: the target class for the clean data
        :param k: number of iterations of the attack
        :param a: the scaling of the gradients used by the attack
        :param random_start: should we start from random perturbation of the clean data
        :param d_min: data min value
        :param d_max: data max value
        :param eot: number of iterations for EOT
        :return: adversarially perturbed data
        """

        model.eval()
        perturbed_data = data.clone()

        perturbed_data.requires_grad = True

        data_max = data + epsilon
        data_min = data - epsilon
        data_max.clamp_(d_min, d_max)
        data_min.clamp_(d_min, d_max)

        if random_start:
            with torch.no_grad():
                perturbed_data.data = data + perturbed_data.uniform_(
                    -1 * epsilon, epsilon)
                perturbed_data.data.clamp_(d_min, d_max)

        for _ in range(k):

            data_grad = 0
            for _ in range(eot):
                output = model(perturbed_data)
                loss = F.cross_entropy(output, target)

                if perturbed_data.grad is not None:
                    perturbed_data.grad.data.zero_()

                loss.backward()
                data_grad += perturbed_data.grad.data
            data_grad /= eot

            with torch.no_grad():
                perturbed_data.data += a * torch.sign(data_grad)
                perturbed_data.data = torch.max(
                    torch.min(perturbed_data, data_max),
                    data_min)
        perturbed_data.requires_grad = False

        return perturbed_data
```

## Run the experiments

### Pre-requisites
- To run the experiments, an access to the ImageNet dataset is required (the folder /home/your-user-name/imagenet should contain the train and val subfolders with the dataset) or see the simple example below with ImageNet samples from foolbox.
- Installation of the foolbox library (please get the version from March 2019 at the farthest), the API was recently modified: <https://github.com/bethgelab/foolbox>
Do the following to install the proper version:
```bash
git clone https://github.com/bethgelab/foolbox.git
cd foolbox
git reset --hard 5191c3a595baadedf0a3659d88b48200024cd534
pip install --editable .
```
- Use PyTorch 1.1 or higher: https://pytorch.org/

### A simple example
We extracted a simple example that shows that adding a uniform noise after the FGSM attack can restore the correct label. The labels are given as numbers from 0 to 999. We use ResNet-18 on 20 ImageNet samples from foolbox. Please, find the Python file attached.

Base test accuracy of the model: 0.95

Accuracy of the model after attack: 0.0

Accuracy of the model after noising: 0.9 (this number can differ because of randomization)

```python
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
    attack = foolbox.attacks.FGSM(model)
    # attack = foolbox.attacks.CarliniWagnerL2Attack(model)
    adversarial_image = attack(image, label)

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
```

#### Examples or results
The Base accuracy is the test accuracy of a "raw" model, for images without ANY distortions (neither adversarial nor from transformations). The adversarial accuracy is the test accuracy only after adding the adversarial perturbations to the images. The recovery accuracy is that test accuracy obtained by applying the FC, CD or noisy transformation after the attack (without attacker being aware of the recovery / compression or the noisy channel). 

| Dataset | # of samples | Attack name | Params | Baseline accuracy | Adversarial accuracy | Recovery accuracy |
|---------|--------------|-------------|--------|---------------|----------------------|-------------------|
| ImageNet | 20 | L2 Carlini-Wagner | iterations=1000 learning_rate=5e-3 initial_const=1e-2 steps=5 | 0.95 | 0.0 | 0.85 |
| ImageNet | 20 | L1 BMI | epsilon=0.3 stepsize=0.05 iterations=10 | 0.95 | 0.0 | 0.85 |
| ImageNet | 20 | L inf FGSM | epsilons=1000 max_epsilon=1 | 0.95 | 0.0 | 0.9 |

##### Results for 1000 samples from ImageNet
The baseline accuracy is the accuracy of the model before any attack, so for inference on images without any perturbations or noise added. Depending on the selected 1000 samples, we have different baseline accuracy of the models. 

![Results of the test accuracy (%) for ImageNet 1000 samples on ResNet-50 after an adversarial attack and recovery with the uniform noise.](NoiseRecoveryAfterAttacks.PNG)

### An example of a run (from a console)

```bash
timestamp=$(date +%Y-%m-%d-%H-%M-%S)

PYTHONPATH=../../../ nohup python3.6 main_adversarial.py --adam_beta2=0.999 --compress_type='STANDARD' --compress_rates 0 --conv_type="STANDARD2D" --conv_exec_type=CUDA --dev_percent=0 --dynamic_loss_scale='TRUE' --epochs=1000 --is_data_augmentation='TRUE' --is_debug='FALSE' --is_dev_dataset='FALSE' --is_progress_bar='FALSE' --learning_rate=0.01 --log_conv_size=FALSE --loss_reduction='ELEMENTWISE_MEAN' --loss_type='CROSS_ENTROPY' --mem_test='FALSE' --memory_size=25 --memory_type='PINNED' --min_batch_size=32 --model_path="no_model" --momentum=0.9 --next_power2='FALSE' --optimizer_type='MOMENTUM' --preserve_energies=100 --sample_count_limit=0 --scheduler_type='ReduceLROnPlateau' --seed=31 --static_loss_scale=1 --stride_type='STANDARD' --tensor_type='FLOAT32' --test_batch_size=32 --use_cuda='TRUE' --visualize='FALSE' --weight_decay=0.0005 --workers=4 --precision_type=FP32 --only_train=FALSE --test_compress_rate='FALSE' --noise_sigmas=0 --start_epsilon=0 --attack_type="RECOVERY" --attack_name="LBFGSAttack" --start_epoch=0 --network_type='ResNet50' --dataset="imagenet" --compress_fft_layer=0 --values_per_channel=0 --recover_type="gauss" --interpolate="const" --step_size=10 >> ${timestamp}.txt 2>&1 &
```
