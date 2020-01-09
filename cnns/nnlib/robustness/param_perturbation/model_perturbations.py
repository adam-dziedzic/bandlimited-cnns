import time
import numpy as np
import sys

from cnns import matplotlib_backend
from cnns.nnlib.utils.exec_args import get_args
from cnns.nnlib.robustness.param_perturbation.utils import get_data_loader
from cnns.nnlib.robustness.param_perturbation.utils import get_accuracy
from cnns.nnlib.robustness.param_perturbation.utils import get_perturbed_fmodel
from cnns.nnlib.robustness.param_perturbation.utils import get_clean_accuracy
from cnns.nnlib.robustness.param_perturbation.sigmas import sigmas1, sigmas2


def compute(args):
    data_loader = get_data_loader(args)

    get_clean_accuracy(args=args, data_loader=data_loader)

    print(f'noise sigma, perturb {args.use_set} accuracy, elapsed time')

    # for noise_sigma in args.noise_sigmas:
    # for noise_sigma in np.linspace(0.0001, 0.01, 100):
    # for noise_sigma in np.linspace(0.0, 0.05, 30):
    # for noise_sigma in sigmas2:
    for noise_sigma in [1e-5, 1e-6, 1e-7]:
        start = time.time()
        args.noise_sigma = noise_sigma
        perturb_fmodel = get_perturbed_fmodel(args)
        perturb_accuracy = get_accuracy(fmodel=perturb_fmodel,
                                        data_loader=data_loader)
        elapsed_time = time.time() - start
        print(args.noise_sigma, ',', perturb_accuracy, ',', elapsed_time)
        sys.stdout.flush()


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
    compute(args)
    print("total elapsed time: ", time.time() - start_time)
