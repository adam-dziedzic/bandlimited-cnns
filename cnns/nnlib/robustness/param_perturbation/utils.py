import numpy as np


def get_adv_images(adversarials, images):
    advs = [a.perturbed for a in adversarials]
    advs = [
        p if p is not None else np.full_like(u, np.nan)
        for p, u in zip(advs, images)
    ]
    advs = np.stack(advs)
    return advs