import torch
import time
import numpy as np
from hessian_eigenthings import compute_hessian_eigenthings
from cnns.nnlib.robustness.fmodel import get_fmodel
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.load_data import get_data


def compute_hessian(args, num_eigens=20):
    fmodel, pytorch_model, from_class_idx_to_label = get_fmodel(args=args)
    train_loader, test_loader, train_dataset, test_dataset, limit = get_data(
        args=args)
    dataloader = test_loader
    if args.is_debug:
        print('dataloader len: ', len(dataloader.dataset))
    loss = torch.nn.functional.cross_entropy

    num_eigenthings = num_eigens  # compute top 20 eigenvalues/eigenvectors

    eigenvals, eigenvecs = compute_hessian_eigenthings(
        pytorch_model, dataloader, loss, num_eigenthings)
    return eigenvals, eigenvecs


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    # args.dataset = 'cifar10'
    args.sample_count_limit = 32
    eigenvals, eigenvecs = compute_hessian(args=args)
    print('eigenvals: ', eigenvals)
    print('total elapsed time: ', time.time() - start_time)
