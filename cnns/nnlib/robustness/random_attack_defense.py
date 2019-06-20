import time
import numpy as np
from cnns.nnlib.utils.exec_args import get_args


if __name__ == "__main__":
    start_time = time.time()
    np.random.seed(31)
    # arguments
    args = get_args()
