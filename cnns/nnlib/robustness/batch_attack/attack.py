#!/usr/bin/env python3

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tfs
import torchvision.datasets as dst
from torch.utils.data import DataLoader
import numpy as np
import time
from cnns.nnlib.robustness.batch_attack.eot_pgd import EOT_PGD
from cnns.nnlib.robustness.batch_attack.raw_pgd import RAW_PGD
from cnns.nnlib.robustness.batch_attack.eot_cw import EOT_CW
from cnns.nnlib.robustness.channels_definition import fft_channel
from cnns.nnlib.robustness.channels_definition import round
from cnns.nnlib.robustness.channels_definition import gauss_noise_torch
from cnns.nnlib.robustness.channels_definition import uniform_noise_torch
from cnns.nnlib.robustness.channels.channels_definition import compress_svd_batch
from cnns.nnlib.robustness.channels_definition import laplace_noise_torch
from cnns.nnlib.utils.complex_mask import get_inverse_hyper_mask
from cnns.nnlib.robustness.channels_definition import subtract_rgb
import cnns.nnlib.pytorch_architecture as models


def attack_eot_pgd(input_v, label_v, net, epsilon=8.0 / 255.0, opt=None):
    eot = EOT_PGD(net=net, epsilon=epsilon, opt=opt)
    adverse_v = eot.eot_batch(images=input_v, labels=label_v)
    diff = adverse_v - input_v
    return adverse_v, diff


def attack_raw_pgd(input_v, label_v, net, epsilon=8.0 / 255.0, opt=None):
    raw_pgd = RAW_PGD(model=net, )


def attack_eot_cw(input_v, label_v, net, c, opt, untarget=True, n_class=10):
    eot = EOT_CW(net=net, c=c, opt=opt, untarget=untarget, n_class=n_class)
    adverse_v = eot.eot_batch(images=input_v, labels=label_v)
    diff = adverse_v - input_v
    return adverse_v, diff


def attack_cw(input_v, label_v, net, c, opt, untarget=True, n_class=10):
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
    optimizer = optim.Adam([w_v], lr=1.0e-3)
    zero_v = torch.tensor([0.0], requires_grad=False).cuda()
    for _ in range(opt.attack_iters):
        net.zero_grad()
        optimizer.zero_grad()
        adverse_v = 0.5 * (torch.tanh(w_v) + 1.0)
        logits = torch.zeros(batch_size, n_class).cuda()
        for i in range(opt.gradient_iters):
            logits += net(adverse_v)
        output = logits / opt.gradient_iters
        # output = logits
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
        diff = adverse_v - input_v
        dist = torch.sum(diff * diff)
        if untarget:
            class_error = torch.sum(torch.max(real - other, zero_v))
        else:
            class_error = torch.sum(torch.max(other - real, zero_v))

        loss = dist + c * class_error
        loss.backward()
        optimizer.step()
    return adverse_v, diff


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
    diff = adverse_v - input_v
    return adverse_v, diff


def attack_rand_fgsm(input_v, label_v, net, epsilon):
    alpha = epsilon / 2
    loss_f = nn.CrossEntropyLoss()
    input_v.requires_grad = True
    adverse = input_v.clone() + alpha * torch.sign(
        torch.FloatTensor(input_v.size()).normal_(0, 1).cuda())
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
    prob = torch.zeros(batch_size, nclass).cuda()
    for i in range(n):
        prob += softmax(net(input_v))
    _, pred = torch.max(prob, 1)
    return pred


def acc_under_attack(dataloader, net, c, attack_f, opt, netAttack=None):
    correct = 0
    tot = 0
    distort = 0.0
    distort_linf = 0.0

    for k, (input, output) in enumerate(dataloader):
        beg = time.time()
        input_v, label_v = input.cuda(), output.cuda()
        # attack
        if netAttack is None:
            netAttack = net
        adverse_v, diff = attack_f(input_v, label_v, netAttack, c, opt)
        # print('min max: ', adverse_v.min().item(), adverse_v.max().item())
        bounds = (0.0, 1.0)
        if opt.channel == 'empty':
            pass
        elif opt.channel == 'gauss':
            adverse_v += gauss_noise_torch(epsilon=opt.noise_epsilon,
                                           images=adverse_v, bounds=bounds)
        elif opt.channel == 'round':
            adverse_v = round(values_per_channel=opt.noise_epsilon,
                              images=adverse_v)
        elif opt.channel == 'fft':
            adverse_v = fft_channel(input=adverse_v,
                                    compress_rate=opt.noise_epsilon)
        elif opt.channel == 'uniform':
            adverse_v += uniform_noise_torch(epsilon=opt.noise_epsilon,
                                             images=adverse_v,
                                             bounds=bounds)
        elif opt.channel == 'laplace':
            adverse_v += laplace_noise_torch(epsilon=opt.noise_epsilon,
                                             images=adverse_v,
                                             bounds=bounds)
        elif opt.channel == 'svd':
            adverse_v = compress_svd_batch(x=adverse_v,
                                           compress_rate=opt.noise_epsilon)
        elif opt.channel == 'inv_fft':
            adverse_v = fft_channel(input=adverse_v,
                                    compress_rate=opt.noise_epsilon,
                                    get_mask=get_inverse_hyper_mask)
        elif opt.channel == 'sub_rgb':
            adverse_v = subtract_rgb(images=adverse_v,
                                     subtract_value=opt.noise_epsilon)
        else:
            raise Exception(f'Unknown channel: {opt.channel}')
        # defense
        net.eval()
        if opt.ensemble == 1:
            _, idx = torch.max(net(adverse_v), 1)
        else:
            idx = ensemble_infer(adverse_v, net, n=opt.ensemble)
        correct += torch.sum(label_v.eq(idx)).item()
        tot += output.numel()
        distort += torch.sum(diff * diff)
        distort_linf += torch.max(torch.abs(diff))

        distort_np = distort.clone().cpu().detach().numpy()
        distort_linf_np = distort_linf.cpu().detach().numpy()

        elapsed = time.time() - beg
        info = ['k', k, 'current_accuracy', correct / tot, 'L2 distortion',
                np.sqrt(distort_np / tot), 'Linf distortion',
                distort_linf_np / tot, 'total_count', tot, 'elapsed time (sec)',
                elapsed]
        # print(','.join([str(x) for x in info]))

        # This is a bit unexpected (shortens computations):
        if opt.limit_batch_number > 0 and k >= opt.limit_batch_number:
            break

    return correct / tot, np.sqrt(distort_np / tot)


def peek(dataloader, net, src_net, c, attack_f, denormalize_layer):
    count, count2, count3 = 0, 0, 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        input_v, label_v = x.cuda(), y.cuda()
        adverse_v = attack_f(input_v, label_v, src_net, c)
        net.eval()
        _, idx = torch.max(net(input_v), 1)
        _, idx2 = torch.max(net(adverse_v), 1)
        idx3 = ensemble_infer2(adverse_v, net)
        count += torch.sum(label_v.eq(idx)).item()
        count2 += torch.sum(label_v.eq(idx2)).item()
        count3 += torch.sum(label_v.eq(idx3)).item()
        less, more = check_in_bound(adverse_v, denormalize_layer)
        print("<0: {}, >1: {}".format(less, more))
        print("Count: {}, Count2: {}, Count3: {}".format(count, count2, count3))
        ok = input("Continue next batch? y/n: ")
        if ok == 'n':
            break


def test_accuracy(dataloader, net):
    net.eval()
    total = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        output = net(x)
        correct += y.eq(torch.max(output, 1)[1]).sum().item()
        total += y.numel()
    acc = correct / total
    return acc


if __name__ == "__main__":

    mod = '0-0'  # mode init noise - inner noise
    bounds = (0, 1)

    if mod == 'trained-1-fft':
        model = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        modelAttack = model
    if mod == '0-0':
        model = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        modelAttack = model
        noiseInit = 0.0
        noiseInner = 0.0
    elif mod == '017-0-test':
        model = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        modelAttack = model
        noiseInit = 0.017
        noiseInner = 0.0
    elif mod == '03-0':
        model = 'rse_0.03_0.0_ady.pth-test-accuracy-0.8574'
        modelAttack = model
        noiseInit = 0.03
        noiseInner = 0.0
    elif mod == '017-0-trained':
        model = 'rse_0.017_0.0_ady.pth-test-accuracy-0.8392'
        # modelAttack = model
        modelAttack = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        noiseInit = 0.017
        noiseInner = 0.0
    elif mod == '2-0':
        model = 'rse_0.2_0.0_ady.pth-test-accuracy-0.8553'
        modelAttack = model
        noiseInit = 0.2
        noiseInner = 0.0
    elif mod == '2-1':
        model = 'rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
        # modelAttack = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        # modelAttack = 'rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
        modelAttack = model
        noiseInit = 0.2
        noiseInner = 0.1
    elif mod == '3-0':
        model = 'rse_0.3_0.0_ady.pth-test-accuracy-0.7618'
        modelAttack = model
        noiseInit = 0.3
        noiseInner = 0.0
    else:
        raise Exception(f'Unknown mod: {mod}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--net', type=str, default='vgg16')
    parser.add_argument('--defense', type=str, default='rse')
    parser.add_argument('--modelIn', type=str,
                        # default='./vgg16/rse_0.2_0.1_ady-ver1.pth',
                        # default='./vgg16/rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
                        default='./vgg16/' + model
                        )
    parser.add_argument('--modelInAttack', type=str,
                        default='./vgg16/' + modelAttack)
    parser.add_argument('--c', type=float, nargs='+',
                        default=[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02,
                                 0.03, 0.04, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4,
                                 0.5, 1.0, 2.0],
                        # default=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.5],
                        # default=[0.001, 0.03, 0.1],
                        # default = '1.0 10.0 100.0 1000.0',
                        # default='0.05,0.1,0.5,1.0,10.0,100.0',
                        )
    parser.add_argument('--noiseInit', type=float, default=noiseInit)
    parser.add_argument('--noiseInner', type=float, default=noiseInner)
    parser.add_argument('--root', type=str, default='data/cifar10-py')
    parser.add_argument('--mode', type=str, default='test')  # peek or test
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--channel', type=str,
                        # default='gauss_torch',
                        # default='round',
                        default='empty',
                        # default='svd',
                        # default='uniform',
                        # default='svd',
                        # default='fft',
                        # default='laplace',
                        # default='inv_fft',
                        # default='sub_rgb'
                        )
    parser.add_argument('--noise_type', type=str,
                        default='standard',
                        # default='backward',
                        )
    parser.add_argument('--attack_iters', type=int, default=300)
    parser.add_argument('--gradient_iters', type=int, default=1)
    parser.add_argument('--eot_sample_size', type=int, default=32)
    parser.add_argument('--limit_batch_number', type=int, default=0,
                        help='If limit > 0, only that # of batches is '
                             'processed. Set this param to 0 to process all '
                             'batches.')
    parser.add_argument('--noise_epsilons', type=float, nargs="+",
                        default=[0.0018],
                        # default=0.3,
                        # default=16,
                        # default=[1.,5.,10.,20.,30.,40.,50.],
                        # default=[2 ** x for x in range(8, 0, -1)],
                        # default=[x for x in range(256)],
                        # default=[43.0], # for RGB reduction
                        # default=[0.0],
                        # default=[0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
                        )

    opt = parser.parse_args()
    opt.bounds = bounds
    print('params: ', opt)
    print('input model: ', opt.modelIn)
    if opt.mode == 'peek' and len(opt.c) != 1:
        print("When opt.mode == 'peek', then only one 'c' is allowed")
        exit(-1)
    netAttack = None
    if opt.net == "vgg16" or opt.net == "vgg16-robust":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.vgg.VGG("VGG16")
        elif opt.defense == "brelu":
            net = models.vgg_brelu.VGG("VGG16", 0.0)
        elif opt.defense == "rse":
            net = models.vgg_rse.VGG("VGG16", opt.noiseInit,
                                     opt.noiseInner,
                                     noise_type='standard')
            # netAttack = net
            netAttack = models.vgg_rse.VGG("VGG16", opt.noiseInit,
                                           opt.noiseInner,
                                           noise_type=opt.noise_type)
            # netAttack = models.vgg_rse.VGG("VGG16", init_noise=0.0,
            #                                inner_noise=0.0,
            #                                noise_type='standard')
    elif opt.net == "resnext":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.resnext.ResNeXt29_2x64d()
        elif opt.defense == "brelu":
            net = models.resnext_brelu.ResNeXt29_2x64d(0)
        elif opt.defense == "rse":
            net = models.resnext_rse.ResNeXt29_2x64d(opt.noiseInit,
                                                     opt.noiseInner)
    elif opt.net == "stl10_model":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.stl10_model.stl10(32)
        elif opt.defense == "brelu":
            # no noise at testing time
            net = models.stl10_model_brelu.stl10(32, 0.0)
        elif opt.defense == "rse":
            net = models.stl10_model_rse.stl10(32, opt.noiseInit,
                                               opt.noiseInner)

    net = nn.DataParallel(net, device_ids=range(1))
    net.load_state_dict(torch.load(opt.modelIn))
    net.cuda()

    if netAttack is not None and id(net) != id(netAttack):
        netAttack = nn.DataParallel(netAttack, device_ids=range(1))
        netAttack.load_state_dict(torch.load(opt.modelInAttack))
        netAttack.cuda()

    loss_f = nn.CrossEntropyLoss()

    if opt.dataset == 'cifar10':
        transform_train = tfs.Compose([
            tfs.RandomCrop(32, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
        ])

        transform_test = tfs.Compose([
            tfs.ToTensor(),
        ])
        data_test = dst.CIFAR10(opt.root, download=True, train=False,
                                transform=transform_test)
    elif opt.dataset == 'stl10':
        transform_train = tfs.Compose([
            tfs.RandomCrop(96, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
        ])
        transform_test = tfs.Compose([
            tfs.ToTensor(),
        ])
        data_test = dst.STL10(opt.root, split='test', download=True,
                              transform=transform_test)
    else:
        print("Invalid dataset")
        exit(-1)
    assert data_test
    dataloader_test = DataLoader(data_test, batch_size=opt.batch_size,
                                 shuffle=False)
    print('Test accuracy {} on clean data for net.'.format(
        test_accuracy(dataloader_test, net)))
    # if netAttack is not None:
    #     print(f'Test accuracy on clean data for netAttack: {test_accuracy(dataloader_test, netAttack)}')

    # attack_f = attack_eot_cw
    # attack_f = attack_eot_pgd
    # attack_f = attack_cw
    attack_f = attack_gauss
    print('attack_f: ', attack_f)

    if opt.mode == 'peek':
        peek(dataloader_test, net, src_net, opt.c[0], attack_f,
             denormalize_layer)
    elif opt.mode == 'test':
        print("#c, noise, test accuracy, L2 distortion, time (sec)")
        for c in opt.c:
            # print('c: ', c)
            for noise in opt.noise_epsilons:
                opt.noise_epsilon = noise
                beg = time.time()
                acc, avg_distort = acc_under_attack(dataloader_test, net, c,
                                                    attack_f, opt,
                                                    netAttack=netAttack)
                timing = time.time() - beg
                print("{}, {}, {}, {}, {}".format(c, noise, acc, avg_distort,
                                                  timing))
                sys.stdout.flush()
    else:
        raise Exception(f'Unknown opt.mode: {opt.mode}')
