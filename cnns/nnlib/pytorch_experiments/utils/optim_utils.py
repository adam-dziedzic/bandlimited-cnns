import torch.optim as optim
from torch.optim.lr_scheduler import \
    ReduceLROnPlateau as ReduceLROnPlateauPyTorch
from torch.optim.lr_scheduler import MultiStepLR

from cnns.nnlib.pytorch_layers.AdamFloat16 import AdamFloat16
from cnns.nnlib.utils.general_utils import OptimizerType
from cnns.nnlib.utils.general_utils import SchedulerType
from cnns.nnlib.utils.general_utils import LossType
from cnns.nnlib.utils.general_utils import LossReduction
import torch


def get_optimizer(args, model):
    params = model.parameters()
    eps = 1e-8
    optimizer_type = args.optimizer_type
    if optimizer_type is OptimizerType.MOMENTUM:
        optimizer = optim.SGD(params, lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif optimizer_type is OptimizerType.ADAM_FLOAT16:
        optimizer = AdamFloat16(params, lr=args.learning_rate, eps=eps)
    elif optimizer_type is OptimizerType.ADAM:
        optimizer = optim.Adam(params, lr=args.learning_rate,
                               betas=(args.adam_beta1, args.adam_beta2),
                               weight_decay=args.weight_decay, eps=eps)
    else:
        raise Exception(f"Unknown optimizer type: {optimizer_type.name}")
    return optimizer


def get_loss_function(args):
    loss_type = args.loss_type
    loss_reduction = args.loss_reduction

    if loss_reduction is LossReduction.MEAN:
        reduction_function = "mean"
    elif loss_reduction is LossReduction.SUM:
        reduction_function = "sum"
    else:
        raise Exception(f"Unknown loss reduction: {loss_reduction}")

    if loss_type is LossType.CROSS_ENTROPY:
        loss_function = torch.nn.CrossEntropyLoss(reduction=reduction_function)
    elif loss_type is LossType.NLL:
        loss_function = torch.nn.NLLLoss(reduction=reduction_function)
    elif loss_type is LossType.MSE:
        loss_function = torch.nn.MSELoss(reduction=reduction_function)
    else:
        raise Exception(f"Unknown loss type: {loss_type}")
    return loss_function


def get_scheduler(args, optimizer):
    scheduler_type = args.scheduler_type
    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if scheduler_type is SchedulerType.ReduceLROnPlateau:
        scheduler = ReduceLROnPlateauPyTorch(optimizer=optimizer, mode='min',
                                             factor=args.schedule_factor,
                                             patience=args.schedule_patience)
    elif scheduler_type is SchedulerType.MultiStepLR:
        scheduler = MultiStepLR(optimizer=optimizer, milestones=[150, 250])
    else:
        raise Exception(f"Unknown scheduler type: {scheduler_type}")
    return scheduler
