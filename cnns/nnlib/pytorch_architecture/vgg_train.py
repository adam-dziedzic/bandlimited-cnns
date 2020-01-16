#!/usr/bin/env python3
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dst
import torchvision.transforms as tfs
from cnns.nnlib.pytorch_architecture.vgg_perturb import VGG as vgg_perturb
from cnns.nnlib.pytorch_architecture.vgg_perturb_weight import \
    VGG as vgg_perturb_weight
from cnns.nnlib.pytorch_architecture.vgg_perturb_conv_fc import \
    VGG as vgg_perturb_conv_fc
from cnns.nnlib.pytorch_architecture.vgg_perturb_conv_bn import \
    VGG as vgg_perturb_conv_bn
from cnns.nnlib.pytorch_architecture.vgg_perturb_fc_bn import \
    VGG as vgg_perturb_fc_bn
from cnns.nnlib.pytorch_architecture.vgg_perturb_fc import \
    VGG as vgg_perturb_fc
from cnns.nnlib.pytorch_architecture.vgg_perturb_bn import \
    VGG as vgg_perturb_bn
from cnns.nnlib.pytorch_architecture.vgg_rse import VGG as vgg_rse
from cnns.nnlib.pytorch_architecture.vgg_perturb_conv import \
    VGG as vgg_perturb_conv
from cnns.nnlib.pytorch_architecture.resnext import ResNeXt29_2x64d
from cnns.nnlib.pytorch_architecture.stl10_model_rse import stl10
from torch.utils.data import DataLoader
import time
import sys


# train one epoch
def train(dataloader, net, loss_f, optimizer):
    net.train()
    beg = time.time()
    total = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        output = net(x)
        lossv = loss_f(output, y)
        lossv.backward()
        optimizer.step()
        correct += y.eq(torch.max(output, 1)[1]).sum().item()
        total += y.numel()
        # print('current accuracy: ', correct/total)
    run_time = time.time() - beg
    return run_time, correct / total


# test and save
def test(dataloader, net, best_acc, opt):
    net.eval()
    total = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        output = net(x)
        correct += y.eq(torch.max(output, 1)[1]).sum().item()
        total += y.numel()
    acc = correct / total
    if acc > best_acc:
        opt.modelOut = opt.modelOutRoot + '-test-accuracy-' + str(acc)
        torch.save(net.state_dict(), opt.modelOut)
        return acc, acc
    else:
        return acc, best_acc


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('LinearNoise') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1 or classname.find('ConvNoise') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--net', type=str, default='vgg16-perturb-conv')
    parser.add_argument('--method', type=str, default="momsgd")
    parser.add_argument('--root', type=str,
                        default="../datasets/cifar-10-batches-py")
    parser.add_argument('--paramNoise', type=float, default=0.0)
    parser.add_argument('--noiseInit', type=float, default=0.2)
    parser.add_argument('--noiseInner', type=float, default=0.1)
    opt = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    opt.modelOutRoot = f"{dir_path}/vgg16/{opt.net}_perturb_{opt.paramNoise}_init_noise_{opt.noiseInit}_inner_noise_{opt.noiseInner}.pth"
    opt.root = dir_path + "/" + opt.root
    print(opt)
    # epochs = [80, 60, 40, 20]
    # epochs = [120, 100, 80, 50]
    epochs = [150, 100, 100, 100]
    # epochs = [1, 1, 1]
    net = None
    if opt.net is None:
        print("opt.net must be specified")
        exit(-1)
    elif opt.net == "vgg16-perturb":
        net = vgg_perturb("VGG16", param_noise=opt.paramNoise)
    elif opt.net == "vgg16-perturb-weight":
        net = vgg_perturb_weight("VGG16", param_noise=opt.paramNoise)
    elif opt.net == "vgg16-perturb-fc":
        net = vgg_perturb_fc("VGG16", param_noise=opt.paramNoise)
    elif opt.net == "vgg16-perturb-bn":
        net = vgg_perturb_bn("VGG16", param_noise=opt.paramNoise)
    elif opt.net == "vgg16-perturb-conv-fc":
        net = vgg_perturb_conv_fc("VGG16", param_noise=opt.paramNoise)
    elif opt.net == "vgg16-perturb-conv-bn":
        net = vgg_perturb_conv_bn("VGG16", param_noise=opt.paramNoise)
    elif opt.net == "vgg16-perturb-conv-fc":
        net = vgg_perturb_fc_bn("VGG16", param_noise=opt.paramNoise)
    elif opt.net == "vgg16-rse":
        net = vgg_rse("VGG16", init_noise=opt.noiseInit,
                      inner_noise=opt.noiseInner)
    elif opt.net == "vgg16-perturb-conv":
        net = vgg_perturb_conv("VGG16", init_noise=opt.noiseInit,
                               inner_noise=opt.noiseInner)
    elif opt.net == "resnext":
        net = ResNeXt29_2x64d()
    elif opt.net == "stl10_model":
        net = stl10(32, opt.noiseInit, opt.noiseInner)
    else:
        raise Exception("Invalid opt.net: {}".format(opt.net))
    net = nn.DataParallel(net, device_ids=range(opt.ngpu))
    net.apply(weights_init)
    net.cuda()
    loss_f = nn.CrossEntropyLoss()
    if opt.dataset == 'cifar10':
        transform_train = tfs.Compose([
            tfs.RandomCrop(32, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor()
        ])

        transform_test = tfs.Compose([
            tfs.ToTensor()
        ])
        data = dst.CIFAR10(opt.root, download=True, train=True,
                           transform=transform_train)
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
        data = dst.STL10(opt.root, split='train', download=False,
                         transform=transform_train)
        data_test = dst.STL10(opt.root, split='test', download=False,
                              transform=transform_test)
    else:
        print("Invalid dataset")
        exit(-1)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=opt.batchSize, shuffle=True,
                            num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=opt.batchSize,
                                 shuffle=False, num_workers=2)
    accumulate = 0
    best_acc = 0
    total_time = 0
    for epoch in epochs:
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=.9,
                              weight_decay=5.0e-4)
        for _ in range(epoch):
            accumulate += 1
            run_time, train_acc = train(dataloader, net, loss_f, optimizer)
            test_acc, best_acc = test(dataloader_test, net, best_acc,
                                      opt)
            total_time += run_time
            print(
                '[Epoch={}] Time:{:.2f}, Train: {:.5f}, Test: {:.5f}, Best: {:.5f}'.format(
                    accumulate, total_time, train_acc, test_acc, best_acc))
            sys.stdout.flush()
        # reload best model
        net.load_state_dict(torch.load(opt.modelOut))
        net.cuda()
        opt.lr /= 10


if __name__ == "__main__":
    main()
