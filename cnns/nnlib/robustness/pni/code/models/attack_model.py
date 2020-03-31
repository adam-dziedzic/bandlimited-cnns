import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
from cnns.nnlib.robustness.batch_attack.attack import attack_cw
from cnns.nnlib.utils.object import Object
from cnns.foolbox.foolbox_3_0_0 import foolbox


class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0,
                 epsilon=0.031, attack_method='pgd',
                 iterations=7):

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id  # this is integer
        self.iterations = iterations

        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd
        elif attack_method == 'cw':
            self.attack_method = self.cw
        elif attack_method == 'boundary':
            self.attack_method = self.boundary_attack

    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader

        if attack_method is not None:
            if attack_method == 'fgsm':
                self.attack_method = self.fgsm
            elif attack_method == 'pgd':
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

    def pgd(self, model, data, target, k=None, a=0.01, random_start=True,
            d_min=0, d_max=1):
        """
        PGD attack.

        :param model:
        :param data:
        :param target:
        :param k: set to either 7 or 40
        :param a:
        :param random_start:
        :param d_min:
        :param d_max:
        :return:
        """
        if k is None:
            k = self.iterations

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

        # diff = data - perturbed_data
        # distort_linf = torch.max(torch.abs(diff))
        # distort_linf_np = distort_linf.cpu().detach().numpy()
        # print('distort_linf_np: ', distort_linf_np)

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

    def boundary_attack(self, net, input_v, label_v, steps=25000):
        net.eval()
        fmodel = foolbox.models.PyTorchModel(net, bounds=(0, 1))
        max_int = sys.maxsize
        attack = foolbox.attacks.BoundaryAttack(
            steps=steps,
            init_attack=foolbox.attacks.LinearSearchBlendedUniformNoiseAttack(directions=max_int, steps=1000),
            # init_attack=foolbox.attacks.BlendedUniformNoiseOnlyAttack(directions=max_int)
        )
        # we skip the second returned value which is the input, and the last one which is success rate
        advs, _, success_adv = attack(fmodel, input_v, label_v, epsilons=None)
        # print('successful adversaries: ', success_adv)
        return advs


def pgd_adapter(input_v, label_v, net, c, opt=None):
    k = opt.attack_iters
    return Attack(dataloader=None, epsilon=c).pgd(
        model=net, data=input_v, target=label_v, a=0.01, k=k)


def boundary_attack_adapter(input_v, label_v, net, c=None, opt=None):
    steps = opt.attack_iters
    return Attack(dataloader=None, epsilon=c).boundary_attack(net=net, input_v=input_v, label_v=label_v, steps=steps)
