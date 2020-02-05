import torch.nn as nn
import torch.nn.functional as F
import torch
from cnns.nnlib.robustness.batch_attack.attack import attack_cw
from cnns.nnlib.utils.object import Object


class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0,
                 epsilon=0.031, attack_method='pgd'):

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id  # this is integer

        if attack_method is 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method is 'pgd':
            self.attack_method = self.pgd
        elif attack_method is 'cw':
            self.attack_method = self.cw

    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader

        if attack_method is not None:
            if attack_method is 'fgsm':
                self.attack_method = self.fgsm
            elif attack_method is 'pgd':
                self.attack_method = self.pgd

    def fgsm(self, model, data, target, data_min=0, data_max=1):

        model.eval()
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()

        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = F.cross_entropy(output, target)

        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward()

        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_data += self.epsilon * sign_data_grad
            # Adding clipping to maintain [min,max] range, default 0,1 for image
            perturbed_data.clamp_(data_min, data_max)

        return perturbed_data

    def pgd(self, model, data, target, k=7, a=0.01, random_start=True,
            d_min=0, d_max=1):

        model.eval()
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()

        perturbed_data.requires_grad = True

        data_max = data + self.epsilon
        data_min = data - self.epsilon
        data_max.clamp_(d_min, d_max)
        data_min.clamp_(d_min, d_max)

        if random_start:
            with torch.no_grad():
                perturbed_data.data = data + perturbed_data.uniform_(
                    -1 * self.epsilon, self.epsilon)
                perturbed_data.data.clamp_(d_min, d_max)

        for _ in range(k):

            output = model(perturbed_data)
            loss = F.cross_entropy(output, target)

            if perturbed_data.grad is not None:
                perturbed_data.grad.data.zero_()

            loss.backward()
            data_grad = perturbed_data.grad.data

            with torch.no_grad():
                perturbed_data.data += a * torch.sign(data_grad)
                perturbed_data.data = torch.max(
                    torch.min(perturbed_data, data_max),
                    data_min)
        perturbed_data.requires_grad = False

        return perturbed_data

    def cw(self, net, input_v, label_v, c=0.01, gradient_iters=1, untarget=True,
           n_class=10, attack_iters=200, channel='empty', noise_epsilon=0):
        opt = Object()
        opt.gradient_iters = gradient_iters
        opt.attack_iters = attack_iters
        opt.channel = channel
        opt.noise_epsilon = noise_epsilon
        opt.ensemble = 1
        opt.limit_batch_number = 0

        return attack_cw(net=net, input_v=input_v, label_v=label_v, c=c,
                         untarget=untarget, n_class=n_class, opt=opt)
