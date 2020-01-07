import numpy as np
from cnns.nnlib.datasets.load_data import get_data


def get_adv_images(adversarials, images):
    advs = [a.perturbed for a in adversarials]
    advs = [
        p if p is not None else np.full_like(u, np.nan)
        for p, u in zip(advs, images)
    ]
    advs = np.stack(advs)
    return advs


def get_data_loader(args):
    train_loader, test_loader, train_dataset, test_dataset, limit = get_data(
        args=args)
    if args.use_set == 'test_set':
        data_loader = test_loader
    elif args.use_set == 'train_set':
        data_loader = train_loader
    else:
        raise Exception("Unknown use set: ", args.use_set)
    print(f"Using set: {args.use_set} for source of data to find "
          f"adversarial examples.")
    return data_loader
