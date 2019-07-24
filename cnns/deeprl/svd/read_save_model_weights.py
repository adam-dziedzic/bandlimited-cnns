import torch
import numpy as np
from cnns.deeprl.pytorch_model import load_model
from cnns.nnlib.utils.exec_args import get_args


def run(args):
    model = load_model(args=args)
    W1 = model.fc1.weight
    W2 = model.fc2.weight
    W3 = model.fc3.weight





if __name__ == '__main__':
    args = get_args()
    prefix = '../dagger_models/'
    policy_file = '2019-07-22-14-35-47-071004_return_-12.745396640610261_train_loss_5.3856566757884196e-06_test_loss_6.146704956887389e-06_.model'
    # policy_file = '2019-07-23-18-55-43-536250_return_-4.858406314762474_train_loss_8.645356267153723e-05_test_loss_1.3776220557029415e-05_.model'
    args.learn_policy_file = prefix + policy_file
    run(args=args)