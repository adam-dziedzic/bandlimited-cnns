#!/usr/bin/env python3

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from cnns.nnlib.robustness.fast_attack.data_saver import DataSaver
from cnns.nnlib.robustness.fast_attack.eot_pgd import EOT_PGD
from cnns.nnlib.robustness.fast_attack.eot_cw import EOT_CW
from cnns.nnlib.robustness.fast_attack.channels import fft_channel
from cnns.nnlib.robustness.fast_attack.channels import round
from cnns.nnlib.robustness.fast_attack.channels import gauss_noise_torch
from cnns.nnlib.robustness.fast_attack.channels import uniform_noise_torch
from cnns.nnlib.robustness.fast_attack.channels import compress_svd_batch
from cnns.nnlib.robustness.fast_attack.channels import laplace_noise_torch
from cnns.nnlib.robustness.fast_attack.complex_mask import \
    get_inverse_hyper_mask
from cnns.nnlib.robustness.fast_attack.channels import subtract_rgb
from cnns.nnlib.robustness.fast_attack.nattack import nattack
from cnns.nnlib.robustness.fast_attack.nattack import \
    iterations as nattack_iterations
from cnns.nnlib.robustness.fast_attack.nattack import npop as nattack_population
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.load_data import get_data
from cnns.nnlib.robustness.foolbox_model import get_fmodel
from cnns.nnlib.datasets.transformations.normalize import Normalize
from cnns.nnlib.datasets.transformations.denormalize import Denormalize
from cnns.nnlib.datasets.transformations.denorm_distance import DenormDistance


def nattack_wrapper(input_v, label_v, net, c=None, opt=None):
    if opt is not None:
        iterations = opt.attack_iters
        population = opt.nattack_population
    else:
        iterations = nattack_iterations
        population = nattack_population

    adv_imgs = torch.zeros_like(input_v)
    device = input_v.device
    for i, (input, label) in enumerate(zip(input_v, label_v)):
        adv_img = nattack(input=input, target=label, model=net,
                          iterations=iterations, population=population)
        if adv_img is not None:
            adv_imgs[i] = adv_img.to(device)
        else:
            adv_imgs[i] = input
    return adv_imgs


def attack_eot_pgd(input_v, label_v, net, epsilon=8.0 / 255.0, opt=None):
    eot = EOT_PGD(net=net, epsilon=epsilon, opt=opt)
    adverse_v = eot.eot_batch(images=input_v, labels=label_v)
    return adverse_v


def attack_eot_cw(input_v, label_v, net, c, opt, untarget=True, n_class=10):
    eot = EOT_CW(net=net, c=c, opt=opt, untarget=untarget, n_class=n_class)
    adverse_v = eot.eot_batch(images=input_v, labels=label_v)
    return adverse_v


def attack_cw(input, label, net, c, args, untarget=True):
    net.eval()
    # net.train()
    index = label.cpu().view(-1, 1)
    batch_size = input.size()[0]
    # one hot encoding
    label_onehot = torch.zeros(batch_size, args.num_classes,
                               requires_grad=False)
    label_onehot.scatter_(dim=1, index=index, value=1)
    label_onehot = label_onehot.to(args.device)
    # Below is ~artanh: http://bit.ly/2MAtsMX that is defined on interval (0,1)
    input_01 = args.denormalizer(input)
    input_01 = torch.clamp(input_01, min=1e-6, max=1.0 - 1e-6)
    w = 0.5 * torch.log((input_01) / (1 - input_01))
    w_v = w.requires_grad_(True)
    optimizer = optim.Adam([w_v], lr=1.0e-3)
    zero_v = torch.tensor([0.0], requires_grad=False).to(args.device)
    for _ in range(args.attack_max_iterations):
        net.zero_grad()
        optimizer.zero_grad()
        adverse_01 = 0.5 * (torch.tanh(w_v) + 1.0)
        adverse_torch = args.normalizer(adverse_01)
        logits = net(adverse_torch)
        if args.gradient_iters > 1:
            for i in range(args.gradient_iters - 1):
                logits += net(adverse_torch)
            output = logits / args.gradient_iters
        else:
            output = logits
        # The logits for the correct class labels.
        real = (torch.max(torch.mul(output, label_onehot), 1)[0])
        # Zero out the logits for the correct classes and even make them much
        # much smaller so that they are not chosen as the other max class.
        # Then from the logits of other classes find the maximum one.
        other = (
            torch.max(
                torch.mul(output, (1 - label_onehot)) - label_onehot * 10000,
                1)[0])
        # The squared L2 loss of the difference between the adversarial
        # example and the input image.
        diff = adverse_01 - input_01
        dist = torch.sum(diff * diff)
        if untarget:
            class_error = torch.sum(torch.max(real - other, zero_v))
        else:
            class_error = torch.sum(torch.max(other - real, zero_v))

        loss = dist + c * class_error
        loss.backward()
        optimizer.step()
    return adverse_torch


def attack_fgsm(input_v, label_v, net, epsilon):
    loss_f = nn.CrossEntropyLoss()
    input_v.requires_grad = True
    adverse = input_v.clone()
    adverse_v = adverse
    outputs = net(input_v)
    loss = loss_f(outputs, label_v)
    loss.backward()
    grad = torch.sign(input_v.grad)
    adverse_v += epsilon * grad
    return adverse_v


def attack_gauss(input_v, label_v, net, epsilon, opt):
    assert input_v.min() >= 0.0 and input_v.max() <= 1.0
    noise = gauss_noise_torch(epsilon=epsilon,
                              images=input_v,
                              bounds=(0, 1))
    adverse_v = input_v + noise
    return adverse_v


def attack_rand_fgsm(input_v, label_v, net, epsilon):
    alpha = epsilon / 2
    loss_f = nn.CrossEntropyLoss()
    input_v.requires_grad = True
    adverse = input_v.clone() + alpha * torch.sign(
        torch.FloatTensor(input_v.size()).normal_(0, 1).to(args.device))
    adverse_v = adverse
    outputs = net(input_v)
    loss = loss_f(outputs, label_v)
    loss.backward()
    grad = torch.sign(input_v.grad)
    adverse_v += (epsilon - alpha) * grad
    return adverse_v


# Ensemble by sum of probability
def ensemble_infer(input_v, net, n=50, nclass=10):
    net.eval()
    batch_size = input_v.size()[0]
    softmax = nn.Softmax()
    prob = torch.zeros(batch_size, nclass).to(args.device)
    for i in range(n):
        prob += softmax(net(input_v))
    _, pred = torch.max(prob, 1)
    return pred


def to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def acc_under_attack(dataloader, net, c, attack_f, args, netAttack=None):
    correct = 0
    tot = 0
    distort_l2 = 0.0
    distort_linf = 0.0
    data_saver = DataSaver(dataset=args.dataset)

    for k, (input, labels) in enumerate(dataloader):
        beg = time.time()
        input, labels = input.to(args.device), labels.to(args.device)
        # attack
        if netAttack is None:
            netAttack = net

        adv = attack_f(input, labels, netAttack, c, args)
        # print('min max adverse: ', adverse.min().item(), adverse.max().item())
        bounds = (args.min, args.max)
        if args.recover_type == 'empty':
            pass
        elif args.recover_type == 'gauss':
            adv += gauss_noise_torch(epsilon=args.noise_epsilon,
                                     images=adv, bounds=bounds)
        elif args.recover_type == 'round':
            adv = round(values_per_channel=args.noise_epsilon,
                        images=adv)
        elif args.recover_type == 'fft':
            adv = fft_channel(input=adv,
                              compress_rate=args.noise_epsilon)
        elif args.recover_type == 'uniform':
            adv += uniform_noise_torch(epsilon=args.noise_epsilon,
                                       images=adv,
                                       bounds=bounds)
        elif args.recover_type == 'laplace':
            adv += laplace_noise_torch(epsilon=args.noise_epsilon,
                                       images=adv,
                                       bounds=bounds)
        elif args.recover_type == 'svd':
            adv = compress_svd_batch(x=adv,
                                     compress_rate=args.noise_epsilon)
        elif args.recover_type == 'inv_fft':
            adv = fft_channel(input=adv,
                              compress_rate=args.noise_epsilon,
                              get_mask=get_inverse_hyper_mask)
        elif args.recover_type == 'sub_rgb':
            adv = subtract_rgb(images=adv,
                               subtract_value=args.noise_epsilon)
        else:
            raise Exception(f'Unknown recover_type: {args.recover_type}')
        # defense
        net.eval()
        adverse_torch = args.normalizer(adv)
        if args.ensemble > 1:
            idx = ensemble_infer(adverse_torch, net, n=50)
        else:
            logits = net(adverse_torch)
            _, idx = torch.max(logits, dim=1)
        correct += torch.sum(labels.eq(idx)).item()
        tot += labels.numel()

        distort_l2 += args.meter(input, adv, norm=2)
        distort_linf += args.meter(input, adv, norm=float('inf'))

        more_info = True
        if more_info:
            elapsed = time.time() - beg

            info = ['k', k, 'current_accuracy', correct / tot, 'L2 distortion',
                    distort_l2 / tot, 'Linf distortion',
                    distort_linf / tot, 'total_count', tot,
                    'elapsed time (sec)', elapsed]
            print(','.join([str(x) for x in info]))
        data_saver.add_data(adv_images=to_numpy(adverse_torch),
                            adv_labels=to_numpy(idx),
                            org_images=to_numpy(input),
                            org_labels=to_numpy(labels))

        if k * args.test_batch_size % 1024 == 0:
            data_saver.save_adv_org()

    data_saver.save_adv_org()

    return correct / tot, distort_l2 / tot, distort_linf / tot


def get_test_accuracy(dataloader, net, args):
    net.eval()
    total = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)
        output = net(x)
        correct += y.eq(torch.max(output, 1)[1]).sum().item()
        total += y.numel()
    acc = correct / total
    return acc


if __name__ == "__main__":
    args = get_args()
    args.save_out = True

    loss_f = nn.CrossEntropyLoss()

    train_loader, test_loader, train_dataset, test_dataset, limit = get_data(
        args=args)
    _, model, _ = get_fmodel(args=args)

    # We need one more dimension for the batch:
    args.mean_array = np.expand_dims(args.mean_array, 0)
    args.std_array = np.expand_dims(args.std_array, 0)

    args.normalizer = Normalize(mean_array=args.mean_array,
                                std_array=args.std_array,
                                device=args.device)
    args.denormalizer = Denormalize(mean_array=args.mean_array,
                                    std_array=args.std_array,
                                    device=args.device)
    args.meter = DenormDistance(mean_array=args.mean_array,
                                std_array=args.std_array,
                                device=args.device)

    test_accuracy = get_test_accuracy(dataloader=test_loader, net=model,
                                      args=args)
    print(f'Test accuracy: {test_accuracy} on clean data for: {args.dataset}')
    # if netAttack is not None:
    #     print(f'Test accuracy on clean data for netAttack: {test_accuracy(dataloader_test, netAttack)}')

    # attack_f = attack_eot_cw
    # attack_f = attack_eot_pgd
    attack_f = attack_cw
    # attack_f = attack_gauss
    # attack_f = nattack_wrapper
    print('attack_f: ', attack_f)

    print(
        "#c, noise, test accuracy, L2 distortion, L inf distortion, time (sec)")
    for c in args.attack_strengths:
        # print('c: ', c)
        for noise in args.noise_epsilons:
            args.noise_epsilon = noise
            beg = time.time()
            acc, avg_L2distort, avg_Linfdistort = acc_under_attack(
                test_loader, model, c, attack_f, args)
            timing = time.time() - beg
            print("{}, {}, {}, {}, {}, {}".format(c, noise, acc,
                                                  avg_L2distort,
                                                  avg_Linfdistort,
                                                  timing))
            sys.stdout.flush()
