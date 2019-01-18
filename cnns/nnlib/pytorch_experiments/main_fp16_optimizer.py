import argparse
import os
import shutil
import time
import socket
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from cnns.nnlib.pytorch_architecture.model_utils import getModelPyTorch
from cnns.nnlib.datasets.cifar import get_cifar
from cnns.nnlib.utils.general_utils import PrecisionType
from cnns.nnlib.utils.exec_args import get_args
import numpy as np
import pathlib
from cnns.nnlib.utils.general_utils import get_log_time

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

args = get_args()

cudnn.benchmark = True

best_prec1 = 0

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.local_rank)

results_folder_name = "results"
results_dir = os.path.join(os.curdir, results_folder_name)
pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval().half()


def main():
    global best_prec1, args

    global_log_file = os.path.join(results_folder_name,
                                   get_log_time() + "-ucr-fcnn.log")
    args_str = args.get_str()
    hostname = socket.gethostname()
    HEADER = "hostname," + str(
        hostname) + ",timestamp," + get_log_time() + "," + str(args_str)
    with open(global_log_file, "a") as file:
        # Write the metadata.
        file.write(HEADER + ",")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.precision_type is PrecisionType.FP16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn " \
                                             "backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.precision_type is PrecisionType.FP16:
            print(
                "Warning:  if --fp16 is not used, static_loss_scale will be "
                "ignored.")

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    #
    # if (args.network_type == "inception_v3"):
    #     crop_size = 299
    #     val_size = 320  # I chose this value arbitrarily, we can adjust.
    # else:
    #     crop_size = 224
    #     val_size = 256
    #
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(crop_size),
    #         transforms.RandomHorizontalFlip(),
    #         # transforms.ToTensor(), Too slow
    #         # normalize,
    #     ]))
    # val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
    #     transforms.Resize(val_size),
    #     transforms.CenterCrop(crop_size),
    # ]))
    args.dataset = str(args.dataset)
    print("args.dataset: ", args.dataset)
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        _, _, train_dataset, test_dataset = get_cifar(args, args.dataset)
    else:
        raise ValueError(f"Unknown dataset: <{args.dataset}>")

    train_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.min_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.min_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler)

    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.network_type))
    #     model = models.__dict__[args.network_type](pretrained=True)
    # else:
    print("=> creating model '{}'".format(args.network_type))
    # model = models.__dict__[args.network_type]()
    model = getModelPyTorch(args=args)
    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()
    if args.precision_type is PrecisionType.FP16:
        model = network_to_half(model)
        model.apply(fix_bn)  # fix batchnorm
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Scale learning rate based on global batch size
    args.lr = args.learning_rate * float(
        args.min_batch_size * args.world_size) / 256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.precision_type is PrecisionType.FP16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale)

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume,
                                        map_location=lambda storage,
                                                            loc: storage.cuda(
                                            args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                # An FP16_Optimizer instance's state dict internally stashes the master params.
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    if args.evaluate:
        test(test_loader, model, criterion)
        return

    with open(global_log_file, "a") as file:
        # Write the header with the names of the columns.
        file.write(
            "\nepoch,train_loss,train_accuracy,test_loss,"
            "test_accuracy,epoch_time,train_time,test_time\n")

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        start_train = time.time()
        train_accuracy, train_loss = train(train_loader, model, criterion,
                                           optimizer, epoch)
        train_time = time.time() - start_train

        if args.prof:
            break
        # evaluate on validation set
        start_test = time.time()
        test_accuracy, test_loss = test(test_loader, model, criterion)
        test_time = time.time() - start_test

        epoch_time = train_time + test_time

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = test_accuracy > best_prec1
            best_prec1 = max(test_accuracy, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.network_type,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        with open(global_log_file, "a") as file:
            file.write(str(epoch) + "," + str(train_loss) + "," + str(
                train_accuracy) + "," + str(test_loss) + "," + str(
                test_accuracy) + "," + str(epoch_time) + "," + str(
                train_time) + "," + str(test_time) + "\n")


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        if args.precision_type is PrecisionType.FP16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)
            if args.precision_type is PrecisionType.FP16:
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        if args.prof:
            if i > 10:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.precision_type is PrecisionType.FP16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        input, target = prefetcher.next()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader),
                args.world_size * args.min_batch_size / batch_time.val,
                args.world_size * args.min_batch_size / batch_time.avg,
                batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    return top1.avg, losses.avg


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(test_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test,[{0}/{1}]\t'
                  'Time,{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed,{2:.3f},({3:.3f})\t'
                  'Loss,{loss.val:.4f},({loss.avg:.4f})\t'
                  'Prec@1,{top1.val:.3f},({top1.avg:.3f})\t'
                  'Prec@5,{top5.val:.3f},({top5.avg:.3f})'.format(
                i, len(test_loader),
                args.world_size * args.min_batch_size / batch_time.val,
                args.world_size * args.min_batch_size / batch_time.avg,
                batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

        input, target = prefetcher.next()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
