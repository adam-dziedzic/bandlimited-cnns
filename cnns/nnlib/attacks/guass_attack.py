from foolbox.attacks.base import Attack
import numpy as np

nprng = np.random.RandomState()
nprng.seed(31)

class GaussAttack(Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, epsilon=0.03,
                 bounds=(0, 1)):
        min_, max_ = bounds
        std = epsilon / np.sqrt(3) * (max_ - min_)
        noise = nprng.normal(scale=std, size=input_or_adv.shape)
        noise = noise.astype(dtype=input_or_adv.dtype)
        image = input_or_adv + noise
        return image