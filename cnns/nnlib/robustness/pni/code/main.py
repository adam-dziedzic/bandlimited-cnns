from __future__ import absolute_import
from __future__ import division

from cnns import matplotlib_backend

print("Using:", matplotlib_backend.backend)

import argparse
import copy
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from cnns.nnlib.robustness.batch_attack.attack import acc_under_attack
from cnns.nnlib.robustness.batch_attack.attack import acc_under_blackbox_attack
from cnns.nnlib.robustness.batch_attack.attack import attack_cw
from cnns.nnlib.robustness.pni.code import models
from cnns.nnlib.robustness.pni.code.models.attack_model import Attack
from cnns.nnlib.robustness.pni.code.models.attack_model import boundary_attack_adapter
from cnns.nnlib.robustness.pni.code.models.attack_model import pgd_adapter
from cnns.nnlib.robustness.pni.code.models.noise_layer import noise_input_layer
from cnns.nnlib.robustness.pni.code.models.nomarlization_layer import \
    Normalize_layer, noise_Normalize_layer
from cnns.nnlib.robustness.pni.code.utils_.utils import AverageMeter, \
    RecorderMeter, time_string, \
    convert_secs2time
from cnns.nnlib.utils.general_utils import get_log_time
from cnns.nnlib.utils.object import Object
from cnns.nnlib.robustness.pni.code.utils_.printing import print_log
from cnns.nnlib.robustness.pni.code.utils_.load_model import resume_from_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

username = os.getlogin()
parser.add_argument('--data_path',
                    default='/home/' + username + '/data/pytorch/cifar10/',
                    type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10',
                             'mnist'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH',
                    # default='lbcnn',
                    # default='noise_resnet20_input',
                    default='noise_resnet20_robust_02',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names)
                    )
# Optimization options
parser.add_argument('--epochs', type=int, default=160,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer', type=str, default='SGD',
                    choices=['SGD', 'Adam'])
parser.add_argument('--batch_size',
                    type=int,
                    # default=10,
                    # default=2560,
                    default=128,
                    help='Batch size.')
parser.add_argument('--learning_rate', type=float,
                    default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=3e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    # default='./save/save_adv_train_cifar10_noise_resnet20_input_160_SGD_train_layerwise_3e-4decay/mode_best.pth.tar',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='',
                    # default='/home/' + username + '/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_0.2.pth.tar',
                    # default='./save/save_adv_train_cifar10_noise_resnet20_input_160_SGD_train_layerwise_3e-4decay/',
                    # default='/home/' + username + '/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/save_adv_train_cifar10_noise_resnet20_input_160_SGD_train_layerwise_3e-4decay/model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--model_only', dest='model_only', action='store_true',
                    help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')

# adversarial training
parser.add_argument('--epoch_delay', type=int, default=5,
                    help='Number of epochs delayed \
                    for starting the adversarial traing')
parser.add_argument('--adv_train',
                    dest='adv_train',
                    action='store_true',
                    help='enable the adversarial training')
parser.add_argument('--adv_eval', dest='adv_eval',
                    action='store_true',
                    help='enable the adversarial evaluation')
parser.add_argument('--attack',
                    default='pgd',
                    type=str,
                    help='name of the attack: pgd, cw, boundary')
parser.add_argument('--attack_eval',
                    dest='attack_eval',
                    action='store_true',
                    help='evaluate the adaptive attack to plot the distortion '
                         'vs accuracy graph')
parser.add_argument('--attack_iters', type=int, nargs='+',
                    default=[100],
                    # default=[25000],
                    help='number of attack iterations')
parser.add_argument('--eot', type=int,
                    default=1,
                    help='number of iterations used to computed averaged gradient')
parser.add_argument('--attack_strengths', type=float, nargs='+',
                    # default=[0.031],
                    default=[0.0, 0.005, 0.01, 0.015, 0.02, 0.022,
                             0.025, 0.028, 0.03, 0.031, 0.032,
                             0.033, 0.034, 0.035, 0.036, 0.037,
                             0.038, 0.039, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4,
                             0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                             1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
                    help='the strengths of the attack (this is the maximum '
                         'L infinity distortions for the PGD attack and the '
                         'c parameter for the Carlini-Wagner L2 attack).')
parser.add_argument('--limit_batch_number', type=int, default=0,
                    help='Used for debugging and tests, limits the batch number for the test set.')

# PNI technique
parser.add_argument('--input_noise',
                    dest='input_noise',
                    action='store_true',
                    help='enable PNI for input, which is right after '
                         'normalization layer')

# RobustNet noise tuning
parser.add_argument('--init_noise', type=float, default=0.2,
                    help='init noise')
parser.add_argument('--inner_noise', type=float, default=0.1,
                    help='inner noise')
parser.add_argument('--noise_type', type=str,
                    default='gauss',
                    help='the type of noise for RobustNet (e.g. gauss or uniform)')

# normalization
parser.add_argument('--normalization',
                    dest='normalization',
                    action='store_true',
                    help='normalize inputs')

##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    # make only devices indexed by #gpu_id visible
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True


###############################################################################
###############################################################################


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path,
                            'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(
        torch.backends.cudnn.version()), log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log')
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.normalization:
        # mean and standard deviation to be used for normalization
        if args.dataset == 'cifar10':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif args.dataset == 'cifar100':
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
        elif args.dataset == 'svhn':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        elif args.dataset == 'mnist':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        elif args.dataset == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            assert False, "Unknow dataset : {}".format(args.dataset)
    else:
        mean = [0.0, 0.0, 0.0]
        std = [0.0, 0.0, 0, 0]

    # Current data-preprocessing does not include the normalization
    imagenet_train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]
    imagenet_test_transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]

    normal_train_transform = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()]
    normal_test_transform = [
        transforms.ToTensor()]

    # if not performing the adversarial training or evalutaion, we append
    # the normalization back to the preprocessing

    if args.normalization:
        if not (args.adv_train or args.adv_eval):
            imagenet_train_transform.append(transforms.Normalize(mean, std))
            imagenet_test_transform.append(transforms.Normalize(mean, std))
            normal_train_transform.append(transforms.Normalize(mean, std))
            normal_test_transform.append(transforms.Normalize(mean, std))

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose(imagenet_train_transform)
        test_transform = transforms.Compose(imagenet_test_transform)
    else:
        train_transform = transforms.Compose(normal_train_transform)
        test_transform = transforms.Compose(normal_test_transform)

    if args.dataset == 'mnist':
        train_data = dset.MNIST(
            args.data_path, train=True, transform=train_transform,
            download=True)
        test_data = dset.MNIST(args.data_path, train=False,
                               transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(
            args.data_path, train=True, transform=train_transform,
            download=True)
        test_data = dset.CIFAR10(
            args.data_path, train=False, transform=test_transform,
            download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(
            args.data_path, train=True, transform=train_transform,
            download=True)
        test_data = dset.CIFAR100(
            args.data_path, train=False, transform=test_transform,
            download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train',
                               transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test',
                              transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(
            args.data_path, split='train', transform=train_transform,
            download=True)
        test_data = dset.STL10(args.data_path, split='test',
                               transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    if args.arch == 'noise_resnet20_robust':
        net_c = models.noise_resnet20_robust(
            num_classes=num_classes,
            init_noise=args.init_noise,
            inner_noise=args.inner_noise,
            noise_type=args.noise_type,
        )
    else:
        # print('keys of models: ', models.__dict__.keys())
        net_c = models.__dict__[args.arch](num_classes=num_classes)
    # For adversarial case, add normalization layer in front of the original network
    if (args.adv_train or args.adv_eval):
        if not args.input_noise:
            if args.normalization:
                net = torch.nn.Sequential(
                    Normalize_layer(mean, std),
                    net_c
                )
            else:
                net = net_c
        else:
            if args.normalization:
                net = torch.nn.Sequential(
                    noise_Normalize_layer(mean, std, input_noise=True),
                    net_c
                )
            else:
                net = torch.nn.Sequential(
                    noise_input_layer(input_noise=True),
                    net_c
                )
    else:
        net = net_c

    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 0:
            # Always use DataParallel (also on a single gpu to easily switch between
            # the 1 or more gpus.
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # separate the model parameters since we want the trainable
    # noise scaling coefficient is free from the weight penalty (weight decay)
    normal_param = [
        param for name, param in net.named_parameters()
        if not 'alpha_' in name
    ]  # this is the parameters do not contain noise scale coefficient

    alpha_param = [
        param for name, param in net.named_parameters()
        if 'alpha_' in name
    ]

    # Initialize the optimizer
    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD([
            {'params': normal_param},
            {'params': alpha_param, 'weight_decay': 0}
        ],
            lr=state['learning_rate'],
            momentum=state['momentum'], weight_decay=state['decay'],
            nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam([
            {'params': normal_param},
            {'params': alpha_param, 'weight_decay': 0}
        ],
            lr=state['learning_rate'],
            weight_decay=state['decay'])


    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop([
            {'params': normal_param},
            {'params': alpha_param, 'weight_decay': 0}
        ],
            lr=state['learning_rate'], alpha=0.99, eps=1e-08, weight_decay=0,
            momentum=0)

    else:
        raise Exception(f"Unknonw optimizer type: {args.optimizer}")

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epochs

    # optionally resume from a checkpoint
    if args.resume:
        recorder, args.start_epoch = resume_from_checkpoint(
            net=net, resume_file=args.resume, log=log, optimizer=optimizer, recorder=recorder,
            fine_tune=args.fine_tune, start_epoch=args.start_epoch)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    # initialize the attacker object
    model_attack = Attack(dataloader=train_loader,
                          attack_method=args.attack,
                          epsilon=args.attack_strengths[0],
                          iterations=args.attack_iters[0])

    if args.evaluate:
        validate(
            test_loader, net, criterion, log,
            attacker=model_attack, adv_eval=args.adv_eval)
        return

    if args.attack_eval:
        if args.attack == 'boundary':
            blackbox_attack_distortion_accuracy(
                dataloader_test=test_loader,
                net=net,
                attack_name=args.attack,
                attack_iters=args.attack_iters,
                attack_strengths=args.attack_strengths,
                args=args
            )
        else:
            attack_distortion_accuracy(
                dataloader_test=test_loader,
                net=net,
                attack_name=args.attack,
                attack_iters=args.attack_iters,
                attack_strengths=args.attack_strengths,
                args=args
            )
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(
                time_string(), epoch, args.epochs,
                need_time, current_learning_rate,
                current_momentum)
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                recorder.max_accuracy(False),
                100 - recorder.max_accuracy(False)), log)

        # delay the adversarial training after the preset number of epochs
        start_adv_train = False
        if epoch >= args.epoch_delay:
            start_adv_train = (True and args.adv_train)

        train_acc, train_los, train_adv_acc, train_adv_los = train(
            train_loader, net, criterion, optimizer, epoch, log,
            attacker=model_attack, adv_train=start_adv_train)

        # evaluate on validation set
        val_acc, val_los, val_pgd_acc, val_pgd_los, val_fgsm_acc, val_fgsm_los = validate(
            test_loader, net, criterion, log,
            attacker=model_attack, adv_eval=args.adv_eval)

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = (val_acc >= recorder.max_accuracy(False))

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(checkpoint_state, is_best,
                        args.save_path, 'checkpoint.pth.tar', log)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

        # ============ TensorBoard logging ============#

        # Log the graidents distribution
        for name, param in net.named_parameters():
            name = name.replace('.', '/')
            writer.add_histogram(name, param.clone().cpu(
            ).data.numpy(), epoch + 1, bins='tensorflow')
            if param.grad is not None:
                writer.add_histogram(name + '/grad',
                                     param.grad.clone().cpu().data.numpy(),
                                     epoch + 1, bins='tensorflow')

        # ## Log the weight and bias distribution
        for name, module in net.named_modules():
            name = name.replace('.', '/')
            class_name = str(module.__class__).split('.')[-1].split("'")[0]

            if hasattr(module, 'alpha_w'):
                if module.alpha_w is not None:
                    if module.pni == 'layerwise':
                        writer.add_scalar(name + '/alpha/',
                                          module.alpha_w.clone().item(),
                                          epoch + 1)
                    elif module.pni == 'channelwise':
                        writer.add_histogram(name + '/alpha/',
                                             module.alpha_w.clone().cpu().data.numpy(),
                                             epoch + 1, bins='tensorflow')

        writer.add_scalar('loss/train_loss', train_los, epoch + 1)
        writer.add_scalar('loss/test_loss', val_los, epoch + 1)
        writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
        writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)

        if args.adv_train:
            writer.add_scalar('loss/adv_train_loss', train_adv_los, epoch + 1)
            writer.add_scalar('accuracy/adv_train_accuracy',
                              train_adv_acc, epoch + 1)
        if args.adv_eval:
            writer.add_scalar('loss/pgd_test_loss', val_pgd_los, epoch + 1)
            writer.add_scalar('accuracy/pgd_test_accuracy',
                              val_pgd_acc, epoch + 1)
            writer.add_scalar('loss/fgsm_test_loss', val_fgsm_los, epoch + 1)
            writer.add_scalar('accuracy/fgsm_test_accuracy',
                              val_fgsm_acc, epoch + 1)

    # ============ TensorBoard logging ============#

    log.close()


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, attacker=None,
          adv_train=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_adv = AverageMeter()
    top1_adv = AverageMeter()
    top5_adv = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            # the copy will be asynchronous with respect to the host.
            # target = target.cuda(async=True)
            target = target.cuda()
            input = input.cuda()

        # compute output for clean data input
        output = model(input)
        loss = criterion(output, target)
        pred_target = output.max(1, keepdim=True)[1].squeeze(-1)

        # perturb data inference
        if adv_train and (attacker is not None):
            model_cp = copy.deepcopy(model)
            perturbed_data = attacker.attack_method(
                model_cp, input, pred_target)
            output_adv = model(perturbed_data)
            loss_adv = criterion(output_adv, target)

            loss = 0.5 * loss + 0.5 * loss_adv

            prec1_adv, prec5_adv = accuracy(
                output_adv.data, target, topk=(1, 5))
            losses_adv.update(loss_adv.item(), input.size(0))
            top1_adv.update(prec1_adv.item(), input.size(0))
            top5_adv.update(prec5_adv.item(), input.size(0))

        # measure accuracy and record the total loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1,
                top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
            top1=top1, top5=top5,
            error1=100 - top1.avg),
        log)

    print_log(
        '  **Adversarial Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
            top1=top1_adv, top5=top5_adv,
            error1=100 - top1_adv.avg),
        log)

    return top1.avg, losses.avg, top1_adv.avg, losses_adv.avg


def validate(val_loader, model, criterion, log, attacker=None, adv_eval=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_pgd = AverageMeter()
    top1_pgd = AverageMeter()
    top5_pgd = AverageMeter()

    losses_fgsm = AverageMeter()
    top1_fgsm = AverageMeter()
    top5_fgsm = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # evaluation for adversarial attack
        if adv_eval and (attacker is not None):
            input.requires_grad = False

            # evaluate the test accuracy under fgsm attack
            attacker.update_params(attack_method='fgsm')
            perturbed_data = attacker.attack_method(model, input, target)
            output_fgsm = model(perturbed_data)
            loss_fgsm = criterion(output_fgsm, target)

            # measure accuracy and record loss
            prec1_fgsm, prec5_fgsm = accuracy(
                output_fgsm.data, target, topk=(1, 5))
            losses_fgsm.update(loss_fgsm.item(), input.size(0))
            top1_fgsm.update(prec1_fgsm.item(), input.size(0))
            top5_fgsm.update(prec5_fgsm.item(), input.size(0))

            input.requires_grad = False

            attacker.update_params(attack_method='pgd')
            perturbed_data = attacker.attack_method(model, input, target)
            output_pgd = model(perturbed_data)
            loss_pgd = criterion(output_pgd, target)

            # measure accuracy and record loss
            prec1_pgd, prec5_pgd = accuracy(
                output_pgd.data, target, topk=(1, 5))
            losses_pgd.update(loss_pgd.item(), input.size(0))
            top1_pgd.update(prec1_pgd.item(), input.size(0))
            top5_pgd.update(prec5_pgd.item(), input.size(0))

    print_log(
        '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
            top1=top1, top5=top5,
            error1=100 - top1.avg), log)

    if adv_eval and (attacker is not None):
        print_log(
            '  **PGD Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
                top1=top1_pgd, top5=top5_pgd,
                error1=100 - top1_pgd.avg), log)
        print_log(
            '  **FGSM Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
                top1=top1_fgsm, top5=top5_fgsm,
                error1=100 - top1_fgsm.avg), log)

    return top1.avg, losses.avg, top1_pgd.avg, losses_pgd.avg, top1_fgsm.avg, losses_fgsm.avg


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    assert len(gammas) == len(
        schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr, mu


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


def select_attack(attack_name, attack_iters=[200], attack_strengths=None):
    if attack_name == 'cw':
        attack_f = attack_cw
        if attack_strengths is None:
            attack_strengths = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02,
                                0.03,
                                0.04, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2,
                                10,
                                100,
                                ]
        attack_iters = attack_iters
    elif attack_name == 'pgd':
        attack_f = pgd_adapter
        # attack_strengths = [0.03]
        # attack_strengths = [0, 0.0001, 0.0005, 0.001, 0.005,
        #                     0.01, 0.05, 0.1]
        if attack_strengths is None:
            attack_strengths = [0.0, 0.005, 0.01, 0.015, 0.02, 0.022,
                                0.025, 0.028, 0.03, 0.031, 0.032,
                                0.033, 0.034, 0.035, 0.036, 0.037,
                                0.038, 0.039, 0.04, 0.05, 0.1]
        # attack_iters = [0, 1, 4, 7, 10, 20, 40, 100, 1000]
        attack_iters = attack_iters
    elif attack_name == 'boundary':
        attack_f = boundary_attack_adapter
        attack_iters = attack_iters
        if attack_strengths is None:
            attack_strengths = [0.0, 0.005, 0.01, 0.015, 0.02, 0.022,
                                0.025, 0.028, 0.03, 0.031, 0.032,
                                0.033, 0.034, 0.035, 0.036, 0.037,
                                0.038, 0.039, 0.04, 0.05, 0.1, 0.3,
                                0.5, 0.8, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0]
        # attack_strengths = None
    else:
        raise Exception(f'Unknown attack: {args.attack}')
    return attack_f, attack_strengths, attack_iters


def print_distortions(distortions, attack_strengths, attack_iter, timing, correct_idx):
    for norm in distortions.keys():
        all_distances = distortions[norm]
        log_time = get_log_time()
        norm_str = str(norm)
        print('max L' + norm_str + ' distance: ', all_distances.max())
        np.save(log_time + norm_str + '_distortions', all_distances)
        np.save(log_time + norm_str + '_correct_idx', correct_idx)
        total_count = len(all_distances)
        for j, c in enumerate(attack_strengths):
            dist_indices = all_distances < c
            adv_idx = np.invert(correct_idx)
            indices = np.logical_and(dist_indices, adv_idx)
            distances = all_distances[indices]
            adv_count = len(distances)
            correct_count = total_count - adv_count
            acc = correct_count / total_count
            if len(distances) == 0:
                mean = 0
                max = 0
                min = 0
                median = 0
            else:
                mean = distances.mean()
                max = distances.max()
                min = distances.min()
                median = np.median(distances)

            # proper printing
            if j == 0:
                header = [
                    "iters",
                    "strength",
                    "test accuracy (%)",
                    "L" + norm_str + " distortion mean",
                    "L" + norm_str + " distortion max",
                    "L" + norm_str + " distortion min",
                    "L" + norm_str + " distortion median",
                    "time (sec)",
                ]
                header_str = ",".join([str(x) for x in header])
                print(header_str)
            results = [
                attack_iter,
                c,
                acc * 100,
                mean,
                max,
                min,
                median,
                timing
            ]
            print(",".join([str(x) for x in results]))
        sys.stdout.flush()


def blackbox_attack_distortion_accuracy(
        dataloader_test,
        net,
        attack_iters=(25000,),
        attack_name='boundary',
        attack_strengths=None,
        args=None):
    attack_f, attack_strengths, attack_iters = select_attack(
        attack_name=attack_name, attack_iters=attack_iters,
        attack_strengths=attack_strengths)

    for i, attack_iter in enumerate(attack_iters):
        opt = Object()
        opt.noise_epsilon = 0.0
        opt.gradient_iters = 1
        opt.attack_iters = attack_iter
        opt.channel = 'empty'
        opt.ensemble = 1
        if args is not None:
            opt.limit_batch_number = args.limit_batch_number
        else:
            opt.limit_batch_number = 0

        beg = time.time()
        acc, distortions, correct_idx = acc_under_blackbox_attack(
            dataloader=dataloader_test,
            c=None,
            net=net,
            attack_f=attack_f,
            opt=opt,
            netAttack=net)
        timing = time.time() - beg
        correct_count = correct_idx.sum()
        print('attack name,', attack_name, ',iters,', attack_iter, ',accuracy,', acc,
              ',correct_count,', correct_count, ',total count,', len(correct_idx), ',time (sec),', timing)
        print_distortions(distortions=distortions, attack_strengths=attack_strengths,
                          attack_iter=attack_iter, timing=timing, correct_idx=correct_idx)


def attack_distortion_accuracy(
        dataloader_test, net,
        attack_iters=[200],
        attack_name='cw',
        attack_strengths=[None],
        args=None):
    attack_f, attack_strengths, attack_iters = select_attack(
        attack_name=attack_name, attack_iters=attack_iters,
        attack_strengths=attack_strengths)
    for i, attack_iter in enumerate(attack_iters):
        for j, c in enumerate(attack_strengths):
            opt = Object()
            opt.noise_epsilon = 0.0
            opt.gradient_iters = 1
            opt.attack_iters = attack_iter
            opt.channel = 'empty'
            opt.ensemble = 1
            opt.eot = args.eot
            if args is not None:
                opt.limit_batch_number = args.limit_batch_number
            else:
                opt.limit_batch_number = 0

            beg = time.time()
            acc, l2_distortion, linf_distortion = acc_under_attack(
                dataloader=dataloader_test,
                net=net,
                c=c,
                attack_f=attack_f,
                opt=opt,
                netAttack=net)
            timing = time.time() - beg
            if i == 0 and j == 0:
                header = [
                    "iters",
                    "strength",
                    "test accuracy (%)",
                    "L2 distortion",
                    "Linf distortion",
                    "time (sec)",
                ]
                header_str = ",".join([str(x) for x in header])
                print(header_str)
            print("{},{},{},{},{},{}".format(
                attack_iter,
                c,
                acc * 100,
                l2_distortion,
                linf_distortion,
                timing))
            sys.stdout.flush()


if __name__ == '__main__':
    main()
