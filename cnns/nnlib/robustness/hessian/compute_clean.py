import torch
import time
import numpy as np
from hessian_eigenthings import compute_hessian_eigenthings
from cnns.nnlib.robustness.fmodel import get_fmodel
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.load_data import get_data
from cnns.nnlib.datasets.pickled import get_pickled_args

def compute_hessian(args, num_eigens=20, file_pickle=None):
    fmodel, pytorch_model, from_class_idx_to_label = get_fmodel(args=args)
    if file_pickle:
       test_loader, test_dataset = get_pickled_args(file=file_pickle, args=args)
    else:
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
    # file_pickle = None
    # file_pickle = '../2019-09-11-20-13-15-755242-adv-images'
    # file_pickle = '../2019-09-11-20-45-04-554033-adv-images'
    # file_pickle = '../2019-09-11-20-45-04-554033-org-images'
    # file_pickle = '../2019-09-11-21-11-34-525829-len-32-org-images'
    file_pickle = '../2019-09-11-21-11-34-525829-len-32-adv-images'
    print('file_pickle: ', file_pickle)
    # arguments
    args = get_args()
    # args.dataset = 'cifar10'
    args.sample_count_limit = 0
    eigenvals, eigenvecs = compute_hessian(args=args, file_pickle=file_pickle)
    print('eigenvals: ', eigenvals)
    with open(file_pickle + '-eigenvals', 'w') as f:
        str_vals = "\n".join([str(val) for val in eigenvals])
        f.write(str_vals)

    print('total elapsed time: ', time.time() - start_time)
