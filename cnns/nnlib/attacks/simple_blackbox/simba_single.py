import torch
import torch.nn.functional as F
from cnns.nnlib.attacks.simple_blackbox.torch_01_range import Ranger


def normalize(x, dataset='imagenet'):
    return Ranger(device=x.device).to_torch(x, dataset=dataset)


def get_probs(model, x, dataset):
    output = model(normalize(x.cuda(), dataset=dataset)).cpu()
    return F.softmax(output, dim=-1).squeeze()


# 20-line implementation of (untargeted) SimBA for single image input
def simba_single(model, x, y, num_iters=10000, epsilon=0.2, dataset='imagenet'):
    n_dims = x.numel()
    perm = torch.randperm(n_dims)
    last_probs = get_probs(model, x, dataset=dataset)
    for i in range(num_iters):
        diff = torch.zeros(n_dims)
        diff[perm[i]] = epsilon
        probs = get_probs(model, (x - diff.view(x.size())).clamp(0, 1),
                          dataset=dataset)
        if probs[y] < last_probs[y]:
            x = (x - diff.view(x.size())).clamp(0, 1)
        else:
            probs = get_probs(model, (x + diff.view(x.size())).clamp(0, 1),
                              dataset=dataset)
            if probs[y] < last_probs[y]:
                x = (x + diff.view(x.size())).clamp(0, 1)
        if probs.argmax() != y:
            return x
        last_probs = probs
    return None
