import torch
import time
import numpy as np
from hessian_eigenthings import compute_hessian_eigenthings
from hessian_eigenthings import HVPOperatorInputs
from hessian_eigenthings import HVPOperatorParams
from cnns.nnlib.robustness.fmodel import get_fmodel
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.datasets.load_data import get_data
from cnns.nnlib.datasets.pickled import get_pickled_args
from torch.utils.data import DataLoader


def compute_hessian(args, num_eigens=20, file_pickle=None,
                    hvp_operator_class=HVPOperatorParams):
    fmodel, pytorch_model, from_class_idx_to_label = get_fmodel(args=args)
    if file_pickle:
        test_loader, test_dataset = get_pickled_args(file=file_pickle,
                                                     args=args)
    else:
        train_loader, test_loader, train_dataset, test_dataset, limit = get_data(
            args=args)
    dataloader = test_loader
    if args.is_debug:
        print('dataloader len: ', len(dataloader.dataset))
    loss = torch.nn.functional.cross_entropy

    num_eigenthings = num_eigens  # compute top 20 eigenvalues/eigenvectors
    model = pytorch_model.eval()

    eigenset = []
    for data_batch, target_batch in dataloader:
        for image, label in zip(data_batch, target_batch):
            dataloader = DataLoader([(image, label)], batch_size=1)
            eigenvals, _ = compute_hessian_eigenthings(
                model=model, dataloader=dataloader, loss=loss,
                num_eigenthings=num_eigenthings,
                hvp_operator_class=hvp_operator_class)
            eigenset.append(eigenvals)
    return eigenset


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)

    # hvp_operator_class = HVPOperatorParams
    hvp_operator_class = HVPOperatorInputs

    # file_pickle = None
    # file_pickle = '../2019-09-11-20-13-15-755242-adv-images'
    # file_pickle = '../2019-09-11-20-45-04-554033-adv-images'
    # file_pickle = '../2019-09-11-20-45-04-554033-org-images'
    # file_pickle = '../2019-09-11-21-11-34-525829-len-32-org-images'
    # file_pickle = '../2019-09-11-21-11-34-525829-len-32-adv-images'
    # file_pickle = '../2019-09-12-09-15-11-050891-len-32-gauss-images'
    # file_pickle = '../2019-09-12-09-15-11-046445-len-32-org-images'
    # file_pickle = '../2019-09-12-09-15-11-040897-len-32-adv-images'
    # file_pickle = '../2019-09-12-09-39-58-229330-len-1-adv_recovered-images'
    # file_pickle = '../2019-09-12-09-39-58-229597-len-1-org_recovered-images'
    # file_pickle = '../2019-09-12-09-39-58-229856-len-1-gauss_recovered-images'
    # file_pickle = '../2019-09-12-15-52-21-871557-len-5-adv-images'
    # file_pickle = '../2019-09-12-15-52-21-874077-len-5-gauss-images'
    file_pickle = '../2019-09-12-15-52-21-873237-len-5-org-images'
    # file_pickle = 'none'
    print('file_pickle: ', file_pickle)
    # arguments
    args = get_args()
    # args.dataset = 'cifar10'
    args.sample_count_limit = 0
    eigenset = compute_hessian(
        args=args, file_pickle=file_pickle,
        hvp_operator_class=hvp_operator_class)
    # eigenset = [[1,2,3,4],[5,6,7,8]]
    print('eigenset len:', len(eigenset))
    eigenset = np.array(eigenset)
    mins = np.min(eigenset, axis=0)
    avgs = np.average(eigenset, axis=0)
    maxs = np.max(eigenset, axis=0)
    deli = ';'
    with open(file_pickle + '-eigenvals-min-avg-max', 'w') as f:
        header = ['eigenvalue_min', 'eigenvalue_avg', 'eigenvalue_max']
        f.write(deli.join(header) + '\n')
        for aggregates in zip(mins, avgs, maxs):
            str_vals = deli.join([str(val) for val in aggregates])
            f.write(str_vals + '\n')

    print('total elapsed time: ', time.time() - start_time)
