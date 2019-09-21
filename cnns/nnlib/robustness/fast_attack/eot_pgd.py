import torch
from torch import nn
import torch.nn.functional as F
import sys


def clip(tensor, min_tensor, max_tensor):
    """Imitate numpy's clip."""
    clipped = torch.max(torch.min(tensor, max_tensor), min_tensor)
    return clipped


class EOT_PGD:
    """
    PGD attack (FGSM with many iterations).
    """

    def __init__(self, net, epsilon, opt,
                 learning_rate=0.1, debug=False, untarget=True):
        self._net = net
        self._epsilon = epsilon
        self._attack_iters = opt.attack_iters
        self._learning_rate = learning_rate
        self._debug = debug
        self._sample_size = opt.eot_sample_size
        if untarget == False:  # if targeted attack
            raise NotImplementedError

    def eot_attack(self, x, y):
        """
        Implementation based on:
        https://github.com/anishathalye/obfuscated-gradients/blob/master/randomization/robustml_attack.py

        :param x: input image
        :param y: input label
        :param net: the network
        :param opt: optimizer
        :param untarget: targeted or untargeted attack
        :param n_class: number of classes
        :return: adversarial examples
        """
        self._net.eval()

        loss_f = nn.CrossEntropyLoss()

        adv = torch.clone(x)
        adv.requires_grad = True

        lower = torch.clamp(adv - self._epsilon, 0, 1)
        upper = torch.clamp(adv + self._epsilon, 0, 1)

        ensemble_y = y.repeat(self._sample_size)

        for i in range(self._attack_iters):
            ensemble_adv = adv.repeat(self._sample_size, 1, 1, 1)
            # adv_un = adv.unsqueeze(dim=0)
            # ensemble_adv = torch.cat([adv_un for _ in range(self._sample_size)], dim=0)
            # ensemble_adv = adv_un

            ensemble_logits = self._net(ensemble_adv)
            # ensemble_probs = F.softmax(ensemble_logits, dim=1)
            ensemble_preds = ensemble_logits.argmax(dim=1)
            print('correct label: ', y)
            print('ensemble preds: ', ensemble_preds)
            print('incorrect preds: %d/%d' % (
            torch.sum(ensemble_preds != y).item(), ensemble_preds.numel()))
            print('ensemble logits: ', ensemble_logits.max(dim=1)[0])
            loss = loss_f(ensemble_logits, ensemble_y)

            loss.backward(adv)

            with torch.no_grad():
                adv += self._learning_rate * adv.grad
                # Zero out the existing gradient.
                # adv.grad.zero_()
                # self._net.zero_grad()
                adv = clip(adv, lower, upper)
                adv.requires_grad = True

            if self._debug:
                print(
                    'attack: step %d/%d, loss = %g (true %d, predicted %s)' % (
                        i + 1, self._attack_iters, loss, y, ensemble_preds),
                    file=sys.stderr
                )

            if y not in ensemble_preds:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                break

        return adv

    def eot_batch(self, images, labels):
        advs = torch.zeros_like(images)
        for i, (x, y) in enumerate(zip(images, labels)):
            adv = self.eot_attack(x, y)
            advs[i] = adv
        return advs
