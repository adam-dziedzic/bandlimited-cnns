import torch
import numpy as np
from cnns.deeprl.pytorch_model import load_model
from cnns.nnlib.utils.exec_args import get_args


def run(args):
    model = load_model(args=args)
    Ws = []
    num_layers = 3

    # Extract the weights from the network.
    for layer in range(num_layers):
        W = model.__getattr__('fc' + str(layer + 1)).weight.data
        Ws.append(W)

    # Extract the singular values from the weight matrices.
    for i, W in enumerate(Ws):
        u, s, v = torch.svd(W)
        s = torch.abs(s)
        s = s.detach().numpy()
        s[::-1].sort()
        output_file = 'svd/' + policy_file + '-' + str(i) + '.csv'
        np.savetxt(fname=output_file, X=s, delimiter=',')


if __name__ == '__main__':
    args = get_args()
    # prefix = 'dagger_models/' + args.env_name + '_models/'
    prefix = 'behave_models/'
    # returns = [-12.7, -10.5, -9.6, -8.7, -7.4, -6.2, -4.8]
    # returns = [1, 100, 200, 500, 1000, 10000]
    returns = [10, 100, 1000, 1500, 2000]
    for return_value in returns:
        # policy_file = 'return_' + str(return_value) + '.model'
        # policy_file = '2019-07-23-18-55-43-536250_return_-4.858406314762474_train_loss_8.645356267153723e-05_test_loss_1.3776220557029415e-05_.model'
        # policy_file = 'saved-model-reacher-v2-' + str(return_value) + '-rolls.model'
        policy_file = 'Ant-v2-' + str(return_value) + '.model'
        args.learn_policy_file = prefix + policy_file
        run(args=args)
