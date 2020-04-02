"""
Source: https://github.com/YyzHarry/ME-Net/blob/master/attack_blackbox.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import sys
from PIL import Image
from cvxpy import *

import foolbox
import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torch.utils.data as Data

from datetime import datetime

from cleverhans.attacks import SPSA
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

from cnns.nnlib.robustness.pni.code import models
from cnns.nnlib.robustness.pni.code.utils_.printing import print_log
from cnns.nnlib.robustness.pni.code.utils_.load_model import resume_from_checkpoint

import warnings

warnings.filterwarnings("ignore")

username = os.getlogin()

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
# Setup
parser.add_argument('--ngpu', type=int, default=1, help='The number of GPUs.')

# Data
parser.add_argument('--batch_size', type=int, default=1, help='The batch size.')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')

# Directory
parser.add_argument('--data_dir', default='/home/' + username + '/data/pytorch/cifar10/', help='data path')

# Attack parameters
parser.add_argument('--attack_type', type=str, default='spsa', help='name of the attack')
parser.add_argument('--epsilon', type=float, default=8, help='The upper bound change of L-inf norm on input pixels')
parser.add_argument('--epsilons', type=float, nargs='+',
                    default=[8.0 / 255, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                    help='The upper bounds on the L-inf norm on adversarial input pixels')
parser.add_argument('--iter', type=int, default=2048, help='The number of iterations for iterative attacks')
parser.add_argument('--cw_conf', type=int, default=20, help='The confidence of adversarial examples for CW attack')
parser.add_argument('--spsa_samples', type=int, default=2048,
                    help='The number of SPSA samples for SPSA attack. '
                         'Number of inputs to evaluate at a single time. '
                         'The true batch size (the number of evaluated inputs for each update) is '
                         '`spsa_samples * spsa_iters`')
parser.add_argument('--spsa_iters', type=int, default=1, help='Number of model evaluations before performing an '
                                                              'update, where each evaluation is on spsa_samples different inputs.')

# Log
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    # default='./save/save_adv_train_cifar10_noise_resnet20_input_160_SGD_train_layerwise_3e-4decay/mode_best.pth.tar',
                    help='Folder to save checkpoints and log.')

# Models
parser.add_argument('--target_model',
                    default='/home/' + username + '/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/robust_net_0.2.pth.tar',
                    type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--source_model',
                    default='/home/' + username + '/code/bandlimited-cnns/cnns/nnlib/robustness/pni/code/save/cifar10_vanilla_resnet20_160_SGD_no_adv_train_vanilla_pure_acc_88_00.pth.tar',
                    type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--target_arch',
                    type=str,
                    default='noise_resnet20_robust_02',
                    choices=model_names,
                    help='target model architecture: ' + ' | '.join(model_names)
                    )
parser.add_argument('--source_arch',
                    type=str,
                    default='vanilla_resnet20',
                    choices=model_names,
                    help='source model architecture: ' + ' | '.join(model_names)
                    )
parser.add_argument('--manual_seed', type=int, default=31, help='manual seed')

# normalization
parser.add_argument('--normalization',
                    dest='normalization',
                    action='store_true',
                    help='normalize inputs')

args = parser.parse_args()

config = {
    'epsilon': args.epsilon / 255.,
    'num_steps': args.iter,
    'step_size': 2.0 / 255,
    'random_start': True,
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data():
    batch = unpickle(args.data_dir + 'cifar-10-batches-py/test_batch')
    data = batch[b'data']
    labels = batch[b'labels']
    return data, labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


class CIFAR10_testset(Data.Dataset):

    def __init__(self, target_transform=None):
        self.target_transform = target_transform
        self.test_data, self.test_labels = get_data()
        self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        img = transform_test(img)
        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    def __len__(self):
        return len(self.test_data)


def transfer_attack(target_model, source_model):
    fmodel = foolbox.models.PyTorchModel(target_model, bounds=(0, 1), num_classes=args.num_classes,
                                         preprocessing=(0, 1))
    fmodel_source = foolbox.models.PyTorchModel(source_model, bounds=(0, 1), num_classes=args.num_classes,
                                                preprocessing=(0, 1))
    attack_criteria = foolbox.criteria.Misclassification()

    if args.attack_type == 'fgsm':
        attack = foolbox.attacks.GradientSignAttack(model=fmodel_source, criterion=attack_criteria)
    elif args.attack_type == 'pgd':
        attack = foolbox.attacks.ProjectedGradientDescentAttack(model=fmodel_source, criterion=attack_criteria)
    else:
        attack = foolbox.attacks.CarliniWagnerL2Attack(model=fmodel_source, criterion=attack_criteria)

    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cpu().numpy()[0], int(targets.cpu().numpy())

        if args.attack_type == 'fgsm':
            adversarial = attack(inputs.astype(np.float32), targets, max_epsilon=config['epsilon'])
        elif args.attack_type == 'pgd':
            adversarial = attack(inputs.astype(np.float32), targets, epsilon=config['epsilon'],
                                 stepsize=config['step_size'], iterations=config['num_steps'])
        else:
            adversarial = attack(inputs.astype(np.float32), targets, max_iterations=config['num_steps'],
                                 confidence=args.cw_conf)
        if adversarial is None:
            adversarial = inputs.astype(np.float32)
        if np.argmax(fmodel.predictions(adversarial)) == targets:
            correct += 1.

        sys.stdout.write("\rTransfer-based black-box %s attack... Acc: %.3f%% (%d/%d)" %
                         (args.attack_type, 100. * correct / (batch_idx + 1), correct, batch_idx + 1))
        sys.stdout.flush()

        print('Accuracy under transfer-based %s attack: %.3f%%' % (args.attack_type, 100. * correct / batch_idx))


def boundary_attack(target_model):
    fmodel = foolbox.models.PyTorchModel(target_model, bounds=(0, 1), num_classes=args.num_classes,
                                         preprocessing=(0, 1))
    attack_criteria = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.BoundaryAttack(model=fmodel, criterion=attack_criteria)

    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cpu().numpy()[0], int(targets.cpu().numpy())
        adversarial = attack(inputs.astype(np.float32), targets, iterations=args.iter, log_every_n_steps=999999)
        if adversarial is None:
            adversarial = inputs.astype(np.float32)
        if np.argmax(fmodel.predictions(adversarial)) == targets:
            correct += 1.

        sys.stdout.write("\rBlack-box Boundary attack... Acc: %.3f%% (%d/%d)" %
                         (100. * correct / (batch_idx + 1), correct, batch_idx + 1))
        sys.stdout.flush()

        print('Accuracy under Boundary attack: %.3f%%' % (100. * correct / batch_idx))


def spsa_attack(target_model):
    # Use tf for evaluation on adversarial data
    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, 3, 32, 32,))
    y_op = tf.placeholder(tf.float32, shape=(1,))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(target_model)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

    # Evaluation on clean data
    correct = 0
    total = 0
    clean_preds_op = tf_model_fn(x_op)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        clean_preds = sess.run(clean_preds_op, feed_dict={x_op: inputs, y_op: targets})
        correct += (np.argmax(clean_preds, axis=1) == targets.numpy()).sum()
        total += len(inputs)
    print('Accuracy on clean data: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

    # Create an SPSA attack
    spsa = SPSA(cleverhans_model, sess=sess)

    for epsilon in args.epsilons:
        spsa_params = {
            # 'eps': config['epsilon'],
            'eps': epsilon,
            'nb_iter': config['num_steps'],
            'clip_min': 0.,
            'clip_max': 1.,
            'spsa_samples': args.spsa_samples,  # in this case, the batch_size is equal to spsa_samples
            'spsa_iters': args.spsa_iters,
        }

        adv_x_op = spsa.generate(x_op, y_op, **spsa_params)
        adv_preds_op = tf_model_fn(adv_x_op)

        # Evaluation against SPSA attacks
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            adv_preds = sess.run(adv_preds_op, feed_dict={x_op: inputs, y_op: targets})
            # print('type of adv_preds: ', type(adv_preds))
            # print('adv_preds shape: ', np.shape(adv_preds))
            # print('adv_preds: ', adv_preds)
            # print('adv_preds argmax only: ', np.argmax(adv_preds, axis=1))
            # print('targets: ', targets)
            # print('targets type: ', type(targets))
            # print('adv_preds argmax == targets: ', np.argmax(adv_preds, axis=1) == targets)
            correct += (np.argmax(adv_preds, axis=1) == targets.numpy()).sum()
            total += len(inputs)

            sys.stdout.write(
                "\rBlack-box SPSA attack... Acc: %.3f%% (%d/%d) (eps:%.3f)" % (
                    100. * correct / total, correct, total, epsilon))
            sys.stdout.flush()

        print('eps,accuracy,correct,total,SPSA attack,%.3f,%.3f,%d,%d' % (
            epsilon, 100. * correct / total, correct, total))


def get_log_time():
    # return time.strftime("%Y-%m-%d-%H-%M-%S", datetime.now())
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


if __name__ == '__main__':
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log_time = get_log_time()
    log = open(os.path.join(args.save_path,
                            'log_seed_{}_{}.txt'.format(args.manual_seed, log_time)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    print_log('Prepare data...', log)
    apply_transforms = [transforms.ToTensor()]
    if args.normalization:
        apply_transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    transform_test = transforms.Compose(apply_transforms)
    test_dataset = CIFAR10_testset(target_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=4)

    # Load model
    # print('keys of models: ', models.__dict__.keys())
    target_model = models.__dict__[args.target_arch](num_classes=args.num_classes)
    source_model = models.__dict__[args.source_arch](num_classes=args.num_classes)

    if torch.cuda.is_available():
        target_model = torch.nn.DataParallel(target_model, device_ids=list(range(args.ngpu)))
        source_model = torch.nn.DataParallel(source_model, device_ids=list(range(args.ngpu)))

    print('Load trained models')
    _, _ = resume_from_checkpoint(net=target_model, resume_file=args.target_model, log=log)
    _, _ = resume_from_checkpoint(net=source_model, resume_file=args.source_model, log=log)

    target_model.eval()
    target_model = target_model.to(device)

    # Baseline
    source_model.eval()
    source_model = source_model.to(device)

    # score-based attack
    if args.attack_type == 'spsa':
        spsa_attack(target_model=target_model)
    # decision-based attack
    elif args.attack_type == 'boundary':
        boundary_attack(target_model=target_model)
    # transfer-based attack
    elif args.attack_type == 'transfer':
        transfer_attack(target_model=target_model, source_model=source_model)
    else:
        raise Exception(f'Unknown attack type: {args.attack_type}')
