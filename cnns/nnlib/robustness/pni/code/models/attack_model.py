import torch.nn as nn
import torch.nn.functional as F
import torch
from cnns.nnlib.robustness.channels_definition import fft_layer


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
           n_class=10, attack_iters=200, channel='empty', noise_epsilon=50):
        net.eval()
        # net.train()
        index = label_v.cpu().view(-1, 1)
        batch_size = input_v.size()[0]
        # one hot encoding
        label_onehot = torch.zeros(batch_size, n_class, requires_grad=False)
        label_onehot.scatter_(dim=1, index=index, value=1)
        label_onehot = label_onehot.cuda()
        # Below is ~artanh: http://bit.ly/2MAtsMX that is defined on interval (0,1)
        w = 0.5 * torch.log((input_v) / (1 - input_v))
        w_v = w.requires_grad_(True)
        optimizer = torch.optim.Adam([w_v], lr=1.0e-3)
        zero_v = torch.tensor([0.0], requires_grad=False).cuda()
        adverse_v = None
        for _ in range(attack_iters):
            net.zero_grad()
            if channel == 'fft_adaptive':
                attack_net = torch.nn.Sequential(
                    fft_layer(compress_rate=noise_epsilon),
                    net
                )
            else:
                attack_net = net
            optimizer.zero_grad()
            adverse_v = 0.5 * (torch.tanh(w_v) + 1.0)
            logits = torch.zeros(batch_size, n_class).cuda()
            for i in range(gradient_iters):
                logits += attack_net(adverse_v)
            output = logits / gradient_iters
            # output = logits
            # The logits for the correct class labels.
            real = (torch.max(torch.mul(output, label_onehot), 1)[0])
            # Zero out the logits for the correct classes and even make them much
            # much smaller so that they are not chosen as the other max class.
            # Then from the logits of other classes find the maximum one.
            other = (
                torch.max(
                    torch.mul(output,
                              (1 - label_onehot)) - label_onehot * 10000,
                    1)[0])
            # The squared L2 loss of the difference between the adversarial
            # example and the input image.
            diff = adverse_v - input_v
            dist = torch.sum(diff * diff)
            if untarget:
                class_error = torch.sum(torch.max(real - other, zero_v))
            else:
                class_error = torch.sum(torch.max(other - real, zero_v))

            loss = dist + c * class_error
            loss.backward()
            optimizer.step()
        return adverse_v
