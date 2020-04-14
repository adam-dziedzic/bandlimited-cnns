import torch
from torch import nn
import torch.nn.functional as F
import sys
from torch import optim


def clip(tensor, min_tensor, max_tensor):
    """Imitate numpy's clip."""
    clipped = torch.max(torch.min(tensor, max_tensor), min_tensor)
    return clipped


class EOT_CW:

    def __init__(self, net, opt, c=0.01, sample_size=32, max_steps=1000,
                 learning_rate=1.0e-3, debug=False, untarget=True, n_class=10):
        """
        :param net: the network
        :param opt: options of the processing
        :param untarget: targeted or untargeted attack
        :param n_class: number of classes
        :param c:
        :param sample_size:
        :param max_steps:
        :param learning_rate:
        :param debug:
        """
        self._net = net
        self._opt = opt
        self._c = c
        self._sample_size = sample_size
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._debug = debug
        self._untarget = untarget
        self._n_class = n_class

    def eot_attack(self, x, y):
        """
        Implementation based on:
        https://github.com/anishathalye/obfuscated-gradients/blob/master/randomization/robustml_attack.py

        :param x: input image
        :param y: input label

        :return: adversarial examples
        """
        self._net.eval()

        # Make vectors from the single element input and label y.
        label_v = y.repeat(self._sample_size)
        input_v = x.repeat(self._sample_size, 1, 1, 1)

        index = label_v.cpu().view(-1, 1)
        # one hot encoding
        label_onehot = torch.zeros(self._sample_size, self._n_class,
                                   requires_grad=False)
        label_onehot.scatter_(dim=1, index=index, value=1)
        label_onehot = label_onehot.cuda()

        zero_v = torch.tensor([0.0], requires_grad=False).cuda()

        w = 0.5 * torch.log((x) / (1 - x))
        w.requires_grad = True
        # Below is ~artanh: http://bit.ly/2MAtsMX that is defined on interval (0,1)
        optimizer = optim.Adam([w], lr=self._learning_rate)
        iter = 0
        # for iter in range(self._opt.attack_iters):
        succ_adv = None

        while True:
            print('iter: ', iter)
            print('c: ', self._c)
            iter += 1
            self._net.zero_grad()
            optimizer.zero_grad()
            adv = 0.5 * (torch.tanh(w) + 1.0)
            ensemble_adv = adv.repeat(self._sample_size, 1, 1, 1)
            logits = self._net(ensemble_adv)
            ensemble_preds = logits.argmax(dim=1)
            print('ensemble_preds: ', ensemble_preds)
            print('max ensemble logits: ', logits.max(dim=1)[0])
            real = (torch.max(torch.mul(logits, label_onehot), 1)[0])
            # Zero out the logits for the correct classes and even make them much
            # much smaller so that they are not chosen as the other max class.
            # Then from the logits of other classes find the maximum one.
            other = (torch.max(
                torch.mul(logits, (1 - label_onehot)) - label_onehot * 10000,
                1)[0])
            # The squared L2 loss of the difference between the adversarial
            # example and the input image.
            diff = adv - input_v
            dist = torch.sum(diff * diff)
            print('dist: ', dist)
            if self._untarget:
                class_error = torch.sum(torch.max(real - other, zero_v))
            else:
                class_error = torch.sum(torch.max(other - real, zero_v))

            loss = dist + self._c * class_error
            loss.backward()
            optimizer.step()

            if y not in ensemble_preds:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                # break
                # print('c: ', self._c)
                self._c /= 2
                succ_adv = adv

        return succ_adv

    def eot_batch(self, images, labels):
        advs = torch.zeros_like(images)
        for i, (x, y) in enumerate(zip(images, labels)):
            adv = self.eot_attack(x, y)
            advs[i] = adv
        return advs