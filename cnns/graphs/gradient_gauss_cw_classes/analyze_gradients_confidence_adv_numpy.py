import numpy as np


def compute():
    file_name_sst = 'cw_0.04_gradients.csv'
    data = np.genfromtxt(file_name_sst, delimiter=',')
    print('data shape: ', data.shape)
    print('data sample: ', data[:5, :])


if __name__ == "__main__":
    compute()
