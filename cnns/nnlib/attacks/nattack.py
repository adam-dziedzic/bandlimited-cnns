import numpy as np
import torch
from foolbox.attacks.base import Attack

npop = 300  # population size
sigma = 0.1  # noise standard deviation
alpha = 0.02  # learning rate
# alpha = 0.001  # learning rate
boxmin = 0
boxmax = 1
boxplus = (boxmin + boxmax) / 2.
boxmul = (boxmax - boxmin) / 2.

epsi = 0.031
epsilon = 1e-30


def softmax(x):
    return np.divide(np.exp(x), np.sum(np.exp(x), -1, keepdims=True))


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (np.log((1 + x) / (1 - x))) * 0.5


means = torch.from_numpy(
    np.array([0.4914, 0.4822, 0.4465]).reshape([1, 3, 1, 1]).astype(
        'float32'))

stds = torch.from_numpy(
    np.array([0.2023, 0.1994, 0.2010]).reshape([1, 3, 1, 1]).astype(
        'float32'))


class Nattack(Attack):

    def __init__(self, model=None, means=means, stds=stds, iterations=500):
        super(Nattack, self).__init__()
        self.model = model
        self.means = means
        self.stds = stds
        self.iterations = iterations

    def __call__(self, input_or_adv, label=None, unpack=True,
                 is_channel_last=False):
        return nattack(input=input_or_adv, target=label, model=self.model,
                       means=self.means, stds=self.stds,
                       is_channel_last=is_channel_last,
                       iterations=self.iterations)


def nattack(input, target, model, means=means, stds=stds, is_channel_last=False,
            iterations=500):
    device = input.device
    means = means.to(device)
    stds = stds.to(device)
    input = (input - means) / stds

    if is_channel_last:
        H, W, C = input.shape
    else:
        C, H, W = input

    modify = np.random.randn(1, C, H, W) * 0.001
    for runstep in range(iterations):
        Nsample = np.random.randn(npop, C, H, W)

        modify_try = modify.repeat(npop, 0) + sigma * Nsample

        newimg = torch_arctanh((input - boxplus) / boxmul)

        if is_channel_last:
            newimg = newimg.transpose(2, 0, 1)
        # print('newimg', newimg,flush=True)

        inputimg = np.tanh(newimg + modify_try) * boxmul + boxplus
        if runstep % 10 == 0:
            realinputimg = np.tanh(newimg + modify) * boxmul + boxplus
            realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
            realclipdist = np.clip(realdist, -epsi, epsi)
            # print('realclipdist :', realclipdist, flush=True)
            realclipinput = realclipdist + (
                    np.tanh(newimg) * boxmul + boxplus)
            l2real = np.sum((realclipinput - (
                    np.tanh(newimg) * boxmul + boxplus)) ** 2) ** 0.5
            # l2real =  np.abs(realclipinput - inputs.numpy())
            print('inputs shape: ', input.shape)
            # outputsreal = model(realclipinput.transpose(0,2,3,1)).data.cpu().numpy()
            input_var = torch.from_numpy(realclipinput.astype('float32')).to(
                device)

            # (input_var - means) / stds)
            # outputsreal = model((input_var - means) / stds).data.cpu().numpy()[0]
            outputsreal = model(input_var).data.cpu().numpy()[0]
            outputsreal = softmax(outputsreal)
            # print(outputsreal)
            print('probs ', np.sort(outputsreal)[-1:-6:-1])
            print('target label ', np.argsort(outputsreal)[-1:-6:-1])
            print('negative_probs ', np.sort(outputsreal)[0:3:1])

            if (np.argmax(outputsreal) != target) and (
                    np.abs(realclipdist).max() <= epsi):
                success = True
                # imsave(folder+classes[targets[0]]+'_'+str("%06d" % batch_idx)+'.jpg',inputs.transpose(1,2,0))
                break
        dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
        clipdist = np.clip(dist, -epsi, epsi)
        clipinput = (clipdist + (
                np.tanh(newimg) * boxmul + boxplus)).reshape(npop, C, H, W)
        target_onehot = np.zeros((1, 10))

        target_onehot[0][target] = 1.
        clipinput = np.squeeze(clipinput)
        clipinput = np.asarray(clipinput, dtype='float32')
        input_var = torch.from_numpy(clipinput).to(device)
        # outputs = model(clipinput.transpose(0,2,3,1)).data.cpu().numpy()
        # outputs = model((input_var - means) / stds).data.cpu().numpy()
        outputs = model(input_var).data.cpu().numpy()
        outputs = softmax(outputs)

        target_onehot = target_onehot.repeat(npop, 0)

        real = np.log((target_onehot * outputs).sum(1) + epsilon)
        other = np.log(
            ((1. - target_onehot) * outputs - target_onehot * 10000.).max(
                1)[0] + epsilon)

        loss1 = np.clip(real - other, 0., 1000)

        Reward = 0.5 * loss1
        # Reward = l2dist

        Reward = -Reward

        A = (Reward - np.mean(Reward)) / (np.std(Reward) + 1e-7)

        modify = modify + (alpha / (npop * sigma)) * (
            (np.dot(Nsample.reshape(npop, -1).T, A)).reshape(C, H, W))
