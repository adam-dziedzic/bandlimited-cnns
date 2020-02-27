#!/usr/bin/env python3
import os
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
from cnns import matplotlib_backend
from cnns.nnlib.robustness.batch_attack.eot_pgd import EOT_PGD
from cnns.nnlib.robustness.batch_attack.raw_pgd import RAW_PGD
from cnns.nnlib.robustness.batch_attack.eot_cw import EOT_CW
from cnns.nnlib.robustness.channels_definition import fft_channel
from cnns.nnlib.robustness.channels_definition import fft_layer
from cnns.nnlib.robustness.channels_definition import round
from cnns.nnlib.robustness.channels_definition import gauss_noise_torch
from cnns.nnlib.robustness.channels_definition import uniform_noise_torch
from cnns.nnlib.robustness.channels.channels_definition import \
    compress_svd_batch
from cnns.nnlib.robustness.channels_definition import laplace_noise_torch
from cnns.nnlib.utils.complex_mask import get_inverse_hyper_mask
from cnns.nnlib.robustness.channels_definition import subtract_rgb
import cnns.nnlib.pytorch_architecture as models
from cnns.nnlib.pytorch_architecture import vgg
from cnns.nnlib.pytorch_architecture import vgg_rse
from cnns.nnlib.pytorch_architecture import vgg_perturb
from cnns.nnlib.pytorch_architecture import vgg_perturb_rse
from cnns.nnlib.pytorch_architecture import vgg_perturb_conv
from cnns.nnlib.pytorch_architecture import vgg_perturb_fc
from cnns.nnlib.pytorch_architecture import vgg_perturb_bn
from cnns.nnlib.pytorch_architecture import vgg_perturb_conv_fc
from cnns.nnlib.pytorch_architecture import vgg_perturb_conv_bn
from cnns.nnlib.pytorch_architecture import vgg_perturb_fc_bn
from cnns.nnlib.pytorch_architecture import vgg_perturb_conv_even
from cnns.nnlib.pytorch_architecture import vgg_perturb_conv_every_2nd
from cnns.nnlib.pytorch_architecture import vgg_perturb_conv_every_3rd
from cnns.nnlib.pytorch_architecture import vgg_perturb_weight
from cnns.nnlib.pytorch_architecture import vgg_rse_perturb
from cnns.nnlib.pytorch_architecture import vgg_rse_perturb_weights
from cnns.nnlib.pytorch_architecture import vgg_rse_unrolled
from cnns.nnlib.pytorch_architecture import vgg_fft

from cnns.nnlib.pytorch_architecture import resnet
from cnns.nnlib.robustness.param_perturbation.utils import perturb_model_params


def attack_eot_pgd(input_v, label_v, net, epsilon=8.0 / 255.0, opt=None):
    eot = EOT_PGD(net=net, epsilon=epsilon, opt=opt)
    adverse_v = eot.eot_batch(images=input_v, labels=label_v)
    return adverse_v


def attack_eot_cw(input_v, label_v, net, c, opt, untarget=True, n_class=10):
    eot = EOT_CW(net=net, c=c, opt=opt, untarget=untarget, n_class=n_class)
    adverse_v = eot.eot_batch(images=input_v, labels=label_v)
    return adverse_v


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
        if opt.channel == 'perturb':
            attack_net = get_perturbed_net(opt=opt)
        elif opt.channel == 'fft_adaptive':
            attack_net = torch.nn.Sequential(
                fft_layer(compress_rate=opt.noise_epsilon),
                net
            )
        else:
            attack_net = net
        optimizer.zero_grad()
        adverse_v = 0.5 * (torch.tanh(w_v) + 1.0)
        logits = torch.zeros(batch_size, n_class).cuda()
        for i in range(opt.gradient_iters):
            logits += attack_net(adverse_v)
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
    return adverse_v


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


def get_perturbed_net(opt):
    perturbed_net, _ = get_nets(opt)
    perturbed_net = perturb_model_params(
        model=perturbed_net, epsilon=opt.noise_epsilon, min=0, max=1)
    perturbed_net.eval()
    return perturbed_net


def acc_under_attack(dataloader, net, c, attack_f, opt, netAttack=None):
    correct = 0
    tot = 0
    distort = 0.0
    distort_linf = 0.0

    for k, (input, output) in enumerate(dataloader):
        # beg = time.time()
        input_v, label_v = input.cuda(), output.cuda()
        # attack
        if netAttack is None:
            netAttack = net

        adverse_v = attack_f(input_v, label_v, netAttack, c, opt)
        diff = adverse_v - input_v
        # print('min max: ', adverse_v.min().item(), adverse_v.max().item())
        bounds = (0.0, 1.0)
        if opt.channel == 'empty':
            pass
        elif opt.channel == 'perturb':
            pass
        elif opt.channel == 'gauss':
            adverse_v += gauss_noise_torch(epsilon=opt.noise_epsilon,
                                           images=adverse_v, bounds=bounds)
        elif opt.channel == 'round':
            adverse_v = round(values_per_channel=opt.noise_epsilon,
                              images=adverse_v)
        elif opt.channel in ('fft', 'fft_adaptive'):
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
            if opt.channel == 'perturb':
                net_infer = get_perturbed_net(opt=opt)
            else:
                net_infer = net
            _, idx = torch.max(net_infer(adverse_v), 1)
        else:
            idx = ensemble_infer(adverse_v, net, n=opt.ensemble)
        correct += torch.sum(label_v.eq(idx)).item()
        tot += output.numel()
        distort += torch.sum(diff * diff)
        distort_linf += torch.max(torch.abs(diff))

        distort_np = distort.clone().cpu().detach().numpy()
        distort_linf_np = distort_linf.cpu().detach().numpy()

        # elapsed = time.time() - beg
        # info = ['k', k, 'current_accuracy', correct / tot, 'L2 distortion',
        #         np.sqrt(distort_np / tot), 'Linf distortion',
        #         distort_linf_np / tot, 'total_count', tot, 'elapsed time (sec)',
        #         elapsed]
        # print(','.join([str(x) for x in info]))

        # This is a bit unexpected (shortens computations):
        if opt.limit_batch_number > 0 and k >= opt.limit_batch_number:
            break

    return correct / tot, np.sqrt(distort_np / tot), distort_linf_np / tot


def peek(dataloader, net, src_net, c, attack_f):
    count, count2, count3 = 0, 0, 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        input_v, label_v = x.cuda(), y.cuda()
        adverse_v = attack_f(input_v, label_v, src_net, c)
        net.eval()
        _, idx = torch.max(net(input_v), 1)
        _, idx2 = torch.max(net(adverse_v), 1)
        idx3 = ensemble_infer(adverse_v, net)
        count += torch.sum(label_v.eq(idx)).item()
        count2 += torch.sum(label_v.eq(idx2)).item()
        count3 += torch.sum(label_v.eq(idx3)).item()
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


def get_nets(opt):
    netAttack = None
    if opt.net == "vgg16" or opt.net == "vgg16-robust":
        if opt.defense in ("plain", "adv", "dd"):
            net = vgg.VGG("VGG16")
        elif opt.defense == "brelu":
            net = models.vgg_brelu.VGG("VGG16", 0.0)
        elif opt.defense == "fft":
            net = models.vgg_fft.VGG("VGG16", compress_rate=opt.compress_rate)
        elif opt.defense == "perturb":
            net = vgg_perturb.VGG("VGG16", param_noise=opt.paramNoise)
            # netAttack = net
            netAttack = vgg_perturb.VGG("VGG16", param_noise=opt.paramNoise)
        elif opt.defense == "perturb-conv":
            net = vgg_perturb_conv.VGG("VGG16", init_noise=opt.noiseInit,
                                       inner_noise=opt.noiseInner)
            # netAttack = net
            netAttack = vgg_perturb_conv.VGG("VGG16", init_noise=opt.noiseInit,
                                             inner_noise=opt.noiseInner)
        elif opt.defense == "perturb-conv-even":
            net = vgg_perturb_conv_even.VGG("VGG16", init_noise=opt.noiseInit,
                                            inner_noise=opt.noiseInner)
            # netAttack = net
            netAttack = vgg_perturb_conv_even.VGG("VGG16",
                                                  init_noise=opt.noiseInit,
                                                  inner_noise=opt.noiseInner)
        elif opt.defense == "perturb-conv-every-2nd":
            net = vgg_perturb_conv_every_2nd.VGG(
                "VGG16",
                init_noise=opt.noiseInit,
                inner_noise=opt.noiseInner)
            # netAttack = net
            netAttack = vgg_perturb_conv_every_2nd.VGG(
                "VGG16",
                init_noise=opt.noiseInit,
                inner_noise=opt.noiseInner)
        elif opt.defense == "perturb-conv-every-3rd":
            net = vgg_perturb_conv_every_3rd.VGG(
                "VGG16",
                init_noise=opt.noiseInit,
                inner_noise=opt.noiseInner)
            # netAttack = net
            netAttack = vgg_perturb_conv_every_3rd.VGG(
                "VGG16",
                init_noise=opt.noiseInit,
                inner_noise=opt.noiseInner)
        elif opt.defense == "perturb-fc":
            net = vgg_perturb_fc.VGG("VGG16", param_noise=opt.paramNoise)
            # netAttack = net
            netAttack = vgg_perturb_fc.VGG("VGG16", param_noise=opt.paramNoise)
        elif opt.defense == "perturb-bn":
            net = vgg_perturb_bn.VGG("VGG16", param_noise=opt.paramNoise)
            # netAttack = net
            netAttack = vgg_perturb_bn.VGG("VGG16", param_noise=opt.paramNoise)
        elif opt.defense == "perturb-conv-fc":
            net = vgg_perturb_conv_fc.VGG("VGG16", param_noise=opt.paramNoise)
            # netAttack = net
            netAttack = vgg_perturb_conv_fc.VGG("VGG16",
                                                param_noise=opt.paramNoise)
        elif opt.defense == "perturb-conv-bn":
            net = vgg_perturb_conv_bn.VGG("VGG16", param_noise=opt.paramNoise)
            # netAttack = net
            netAttack = vgg_perturb_conv_bn.VGG("VGG16",
                                                param_noise=opt.paramNoise)
        elif opt.defense == "perturb-fc-bn":
            net = vgg_perturb_fc_bn.VGG("VGG16", param_noise=opt.paramNoise)
            # netAttack = net
            netAttack = vgg_perturb_fc_bn.VGG("VGG16",
                                              param_noise=opt.paramNoise)
        elif opt.defense == "perturb-weight":
            net = vgg_perturb_weight.VGG("VGG16",
                                         param_noise=opt.paramNoise)
            # netAttack = net
            netAttack = vgg_perturb_weight.VGG("VGG16",
                                               param_noise=opt.paramNoise)
        elif opt.defense == "perturb-rse":
            net = vgg_perturb_rse.VGG("VGG16",
                                      param_noise=opt.paramNoise)
            # netAttack = net
            netAttack = vgg_perturb_rse.VGG("VGG16",
                                            param_noise=opt.paramNoise)
        elif opt.defense == "rse":
            net = vgg_rse.VGG("VGG16", opt.noiseInit,
                              opt.noiseInner,
                              noise_type='standard')
            # netAttack = net
            netAttack = models.vgg_rse.VGG("VGG16", opt.noiseInit,
                                           opt.noiseInner,
                                           noise_type=opt.noise_type)
            # netAttack = models.vgg_rse.VGG("VGG16", init_noise=0.0,
            #                                inner_noise=0.0,
            #                                noise_type='standard')
        elif opt.defense == "rse-non-adaptive":
            net = vgg_rse.VGG("VGG16", opt.noiseInit,
                              opt.noiseInner,
                              noise_type='standard')
            # netAttack = net
            netAttack = models.vgg_rse.VGG("VGG16",
                                           init_noise=0.0,
                                           inner_noise=0.0,
                                           noise_type='standard')
        elif opt.defense == "rse-unrolled":
            net = vgg_rse_unrolled.VGG("VGG16", opt.noiseInit,
                                       opt.noiseInner,
                                       noise_type='standard')
            # netAttack = net
            netAttack = vgg_rse_unrolled.VGG("VGG16", opt.noiseInit,
                                             opt.noiseInner,
                                             noise_type=opt.noise_type)
        elif opt.defense == "rse-perturb":
            net = vgg_rse_perturb.VGG("VGG16", init_noise=opt.noiseInit,
                                      inner_noise=opt.noiseInner,
                                      param_noise=opt.paramNoise,
                                      noise_type='standard')
            netAttack = vgg_rse_perturb.VGG("VGG16", init_noise=opt.noiseInit,
                                            inner_noise=opt.noiseInner,
                                            param_noise=opt.paramNoise,
                                            noise_type='standard')
        elif opt.defense == "rse-perturb-weights":
            net = vgg_rse_perturb_weights.VGG("VGG16", init_noise=opt.noiseInit,
                                              inner_noise=opt.noiseInner,
                                              param_noise=opt.paramNoise,
                                              noise_type='standard')
            netAttack = vgg_rse_perturb_weights.VGG("VGG16",
                                                    init_noise=opt.noiseInit,
                                                    inner_noise=opt.noiseInner,
                                                    param_noise=opt.paramNoise,
                                                    noise_type='standard')
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
    elif opt.net == 'resnet18':
        net = resnet.ResNet18()
    else:
        raise Exception(f"Unknown opt.net: {opt.net}")

    # get_parameter_stats(net)

    try:
        gpus = os.environ['CUDA_VISIBLE_DEVICES']
        gpus = list(range(len(gpus.split(','))))
    except KeyError as e:
        # print('error: ', e)
        gpus = [0]

    net = nn.DataParallel(net, device_ids=gpus)
    net.load_state_dict(torch.load(opt.modelIn))
    net.cuda()

    if netAttack is not None and id(net) != id(netAttack):
        netAttack = nn.DataParallel(netAttack, device_ids=range(1))
        netAttack.load_state_dict(torch.load(opt.modelInAttack))
        netAttack.cuda()

    return net, netAttack


def get_dataloader(opt):
    if opt.dataset == 'cifar10':
        opt.root = 'data/cifar10-py'
        transform_train = tfs.Compose([
            tfs.RandomCrop(32, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
        ])

        transform_test = tfs.Compose([
            tfs.ToTensor(),
        ])
        data_test = dst.CIFAR10(opt.root,
                                download=True,
                                train=False,
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
    return dataloader_test


def set_model_settings(opt):
    net_mode = opt.net_mode  # mode init noise - inner noise
    noiseInit = 0.0
    noiseInner = 0.0
    paramNoise = 0.0
    if net_mode == 'trained-1-fft':
        modelPath = 'vgg16/rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        modelAttack = modelPath
        net = 'vgg16'
    elif net_mode == 'vgg-fft-80':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_vgg16_fft_compress_rate_80.0.pth-test-accuracy-0.8505'
        modelAttack = modelPath
        net = 'vgg16'
    elif net_mode == 'vgg-fft-80-non-adaptive':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_vgg16_fft_compress_rate_80.0.pth-test-accuracy-0.8505'
        modelAttack = 'vgg16/rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        net = 'vgg16'
    elif net_mode == '0-0':
        modelPath = 'vgg16/rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        # modelPath = 'vgg16/rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
        modelAttack = modelPath
        net = 'vgg16'
    elif net_mode == '017-0-test':
        modelPath = 'vgg16/rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        modelAttack = modelPath
        noiseInit = 0.017
        net = 'vgg16'
    elif net_mode == '03-0':
        modelPath = 'vgg16/rse_0.03_0.0_ady.pth-test-accuracy-0.8574'
        modelAttack = modelPath
        noiseInit = 0.03
        net = 'vgg16'
    elif net_mode == '017-0-trained':
        modelPath = 'vgg16/rse_0.017_0.0_ady.pth-test-accuracy-0.8392'
        # modelAttack = model
        modelAttack = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        noiseInit = 0.017
        net = 'vgg16'
    elif net_mode == '2-0':
        modelPath = 'vgg16/rse_0.2_0.0_ady.pth-test-accuracy-0.8553'
        modelAttack = modelPath
        noiseInit = 0.2
        net = 'vgg16'
    elif net_mode == '2-1':
        modelPath = 'vgg16/rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
        # modelPath = 'vgg16/rse_0.2_0.1_ady.pth-test-accuracy-0.8516'
        modelAttack = modelPath
        noiseInit = 0.2
        noiseInner = 0.1
        net = 'vgg16'
    elif net_mode == '2-1-non-adaptive':
        modelPath = 'vgg16/rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
        # modelPath = 'vgg16/rse_0.2_0.1_ady.pth-test-accuracy-0.8516'
        modelAttack = 'vgg16/rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        noiseInit = 0.2
        noiseInner = 0.1
        net = 'vgg16'
    elif net_mode == '3-0':
        modelPath = 'vgg16/rse_0.3_0.0_ady.pth-test-accuracy-0.7618'
        modelAttack = modelPath
        noiseInit = 0.3
        net = 'vgg16'
    elif net_mode == 'perturb-0.01-model-0.0':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351'
        modelAttack = modelPath
        paramNoise = 0.01
        net = 'vgg16'
    elif net_mode == 'perturb-0.02-model-0.0':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351'
        modelAttack = modelPath
        paramNoise = 0.02
        net = 'vgg16'
    elif net_mode == 'perturb-0.06-model-0.0':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351'
        modelAttack = modelPath
        paramNoise = 0.06
        net = 'vgg16'
    elif net_mode == 'perturb-0.07-model-0.0':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351'
        modelAttack = modelPath
        paramNoise = 0.07
    elif net_mode == 'perturb-0.03-model-0.0':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351'
        modelAttack = modelPath
        paramNoise = 0.03
        net = 'vgg16'
    elif net_mode == 'perturb-0.04-model-0.0':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351'
        modelAttack = modelPath
        paramNoise = 0.04
        net = 'vgg16'
    elif net_mode == 'perturb-0.045-model-0.0':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351'
        modelAttack = modelPath
        paramNoise = 0.045
        net = 'vgg16'
    elif net_mode == 'perturb-0.05-model-0.0':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351'
        modelAttack = modelPath
        paramNoise = 0.05
        net = 'vgg16'
    elif net_mode == 'perturb-0.01':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.01.pth-test-accuracy-0.9264'
        modelAttack = modelPath
        paramNoise = 0.01
        net = 'vgg16'
    elif net_mode == 'perturb-0.0-model-0.01':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.01.pth-test-accuracy-0.9264'
        modelAttack = modelPath
        paramNoise = 0.0
        net = 'vgg16'
    elif net_mode == 'perturb-0.02':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.02.pth-test-accuracy-0.8943'
        modelAttack = modelPath
        paramNoise = 0.02
        net = 'vgg16'
    elif net_mode == 'perturb-0.0-model-0.02':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.02.pth-test-accuracy-0.8943'
        modelAttack = modelPath
        paramNoise = 0.0
        net = 'vgg16'
    elif net_mode == 'perturb-0.03':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.03.pth-test-accuracy-0.8465'
        modelAttack = modelPath
        paramNoise = 0.03
        net = 'vgg16'
    elif net_mode == 'perturb-0.0-model-0.03':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.03.pth-test-accuracy-0.8465'
        modelAttack = modelPath
        paramNoise = 0.0
        net = 'vgg16'
    elif net_mode == 'perturb-0.035':
        modelPath = '../../pytorch_architecture/vgg16/rse_perturb_0.035.pth-test-accuracy-0.8162'
        modelAttack = modelPath
        paramNoise = 0.035
        net = 'vgg16'
    elif net_mode == 'perturb-0.0-model-0.035':
        modelPath = '../../pytorch_architecture/vgg16/rse_perturb_0.035.pth-test-accuracy-0.8162'
        modelAttack = modelPath
        paramNoise = 0.0
        net = 'vgg16'
    elif net_mode == 'perturb-0.04':
        modelPath = '../../pytorch_architecture/vgg16/rse_perturb_0.04.pth-test-accuracy-0.7866'
        modelAttack = modelPath
        paramNoise = 0.04
        net = 'vgg16'
    elif net_mode == 'perturb-0.045':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.045.pth-test-accuracy-0.7504'
        modelAttack = modelPath
        paramNoise = 0.045
        net = 'vgg16'
    elif net_mode == 'perturb-0.05':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.05.pth-test-accuracy-0.7002'
        modelAttack = modelPath
        paramNoise = 0.05
        net = 'vgg16'
    elif net_mode == 'perturb-0.06':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.06.pth-test-accuracy-0.6429'
        modelAttack = modelPath
        paramNoise = 0.06
        net = 'vgg16'
    elif net_mode == 'perturb-0.07':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.07.pth-test-accuracy-0.5801'
        modelAttack = modelPath
        paramNoise = 0.07
        net = 'vgg16'
    elif net_mode == 'perturb-0.1':
        modelPath = '../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.1.pth-test-accuracy-0.4319'
        modelAttack = modelPath
        paramNoise = 0.1
        net = 'vgg16'
    elif net_mode == 'resnet18':
        modelPath = '../../pytorch_experiments/models/saved-model-2020-01-09-06-14-54-239467-dataset-cifar10-preserve-energy-100.0-compress-rate-0.0-test-loss-0.009636239625141025-test-accuracy-93.27-0-1-normalized-images.model'
        modelAttack = modelPath
        net = 'resnet18'
    elif net_mode == 'custom':
        opt.modelInAttack = opt.modelIn
        return
    else:
        raise Exception(f'Unknown opt.net_mode: {opt.net_mode}')

    opt.modelIn = modelPath
    opt.modelInAttack = modelAttack
    opt.noiseInit = noiseInit
    opt.noiseInner = noiseInner
    opt.paramNoise = paramNoise
    opt.net = net


def get_parameter_stats(model):
    stds = []
    for param in model.parameters():
        x = param.data
        std = torch.std(x).item()
        min = torch.min(x).item()
        print(list(x.shape), std, min)
        stds.append(std)
    print(','.join([str(x) for x in stds]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_mode',
                        type=str,
                        default='custom')
    parser.add_argument('--defense', type=str,
                        # default='rse-perturb',
                        # default='rse',
                        default='perturb-conv',
                        # default='plain',
                        # default='perturb',
                        )
    parser.add_argument('--c', type=float, nargs='+',
                        # default=[0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02,
                        #          0.03, 0.04, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4,
                        #          0.5, 1.0, 2.0, 10.0, 100.0],
                        default=[0.01],
                        # default=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.5],
                        # default=[0.001, 0.03, 0.1],
                        # default = '1.0 10.0 100.0 1000.0',
                        # default='0.05,0.1,0.5,1.0,10.0,100.0',
                        )
    parser.add_argument('--channel', type=str,
                        default='empty',
                        # default='gauss_torch',
                        # default='round',
                        # default='svd',
                        # default='uniform',
                        # default='svd',
                        # default='fft',
                        # default='laplace',
                        # default='inv_fft',
                        # default='sub_rgb'
                        # default='perturb',
                        )
    parser.add_argument('--attack_iters', type=int,
                        # default=300,
                        default=1,
                        )
    parser.add_argument('--limit_batch_number', type=int, default=0,
                        help='If limit > 0, only that # of batches is '
                             'processed. Set this param to 0 to process all '
                             'batches.')
    parser.add_argument('--noise_epsilons', type=float, nargs="+",
                        default=np.linspace(0.09, 0.01, 100),
                        # default=[0.0018, 0.03],
                        # default=[0.0032],
                        # default=[0.0],
                        # default=0.3,
                        # default=16,
                        # default=[1.,5.,10.,20.,30.,40.,50.],
                        # default=[2 ** x for x in range(8, 0, -1)],
                        # default=[x for x in range(256)],
                        # default=[43.0], # for RGB reduction
                        # default=[0.0],
                        # default=[0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
                        )
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10')
    parser.add_argument('--mode', type=str, default='test')  # peek or test
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--batch_size', type=int,
                        default=3584,
                        # default=256,
                        # default=1024,
                        # default=32,
                        )
    parser.add_argument('--noise_type', type=str,
                        default='standard',
                        # default='backward',
                        )
    parser.add_argument('--gradient_iters', type=int, default=1)
    parser.add_argument('--eot_sample_size', type=int, default=32)
    parser.add_argument('--noiseInit', type=float, default=0.0)
    parser.add_argument('--noiseInner', type=float, default=0.0)
    parser.add_argument('--paramNoise', type=float, default=0.0)
    parser.add_argument('--compress_rate', type=float, default=80.0)
    parser.add_argument('--net', type=str, default='vgg16')
    parser.add_argument('--modelIn', type=str,
                        # default='../../pytorch_architecture/vgg16/saved_model_rse_perturb_0.0.pth-test-accuracy-0.9351',
                        default='../../pytorch_architecture/vgg16/saved_model_vgg16-perturb-conv_perturb_0.0_init_noise_0.0_inner_noise_0.0.pth-test-accuracy-0.9384')
    opt = parser.parse_args()
    opt.bounds = (0, 1)

    set_model_settings(opt)

    print('params: ', opt)
    print('input model: ', opt.modelIn)
    if opt.mode == 'peek' and len(opt.c) != 1:
        print("When opt.mode == 'peek', then only one 'c' is allowed")
        exit(-1)

    net, netAttack = get_nets(opt=opt)

    loss_f = nn.CrossEntropyLoss()

    dataloader_test = get_dataloader(opt=opt)

    print('Test accuracy {} on clean data for net.'.format(
        test_accuracy(dataloader_test, net)))
    # if netAttack is not None:
    #     print(f'Test accuracy on clean data for netAttack: {test_accuracy(dataloader_test, netAttack)}')

    # attack_f = attack_eot_cw
    # attack_f = attack_eot_pgd
    attack_f = attack_cw
    # attack_f = attack_gauss
    print('attack_f: ', attack_f)

    if opt.mode == 'peek':
        peek(dataloader_test, net, net, opt.c[0], attack_f)
    elif opt.mode == 'test':
        print("#c, noise, test accuracy, L2 distortion, time (sec)")
        for c in opt.c:
            # print('c: ', c)
            for noise in opt.noise_epsilons:
                opt.noise_epsilon = noise
                beg = time.time()
                acc, l2_dist, linf_dist = acc_under_attack(dataloader_test, net, c,
                                                    attack_f, opt,
                                                    netAttack=netAttack)
                timing = time.time() - beg
                print("{}, {}, {}, {}, {}".format(c, noise, acc, l2_dist,
                                                  timing))
                sys.stdout.flush()
    else:
        raise Exception(f'Unknown opt.mode: {opt.mode}')
