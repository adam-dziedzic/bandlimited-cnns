from torch.utils.data import Dataset
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from cnns.nnlib.utils.general_utils import MemoryType
from torch.utils.data import DataLoader
# from cnns.nnlib.datasets.transformations.to_tensor import ToTensorWithType
import torch


class RolloutsDataset(Dataset):

    def __init__(self, observations, actions, transform=None):
        self.observations = observations
        self.actions = actions
        self.transform = transform

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = self.observations[idx]
        action = self.actions[idx]

        if self.transform:
            observation = self.transform(observation)
            action = self.transform(action)

        return observation, action


def read_data(filename):
    import os
    print('current dir: ', os.getcwd())
    with open(file=filename, mode="rb") as f:
        data = pickle.load(file=f)
    observations = data['observations']
    actions = np.squeeze(data['actions'])
    return observations, actions


def set_args(args):
    use_cuda = args.use_cuda
    num_workers = args.workers
    pin_memory = False
    if args.memory_type is MemoryType.PINNED:
        pin_memory = True
    if use_cuda:
        kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    else:
        kwargs = {'num_workers': num_workers}
    args.in_channels = 1  # number of channels in the input data
    args.out_channels = None
    args.signal_dimension = 1
    return kwargs


def to_tensor(ndarray, dtype):
    return torch.from_numpy(ndarray).to(dtype)


def get_rollouts_dataset(args):
    obs, actions = read_data(args.rollout_file)

    sample_count = args.sample_count_limit
    if sample_count > 0:
        obs, actions = obs[:sample_count], actions[:sample_count]

    args.input_size = obs.shape[1]
    args.output_size = actions.shape[1]

    print('input and output sizes: ', args.input_size, ' ', args.output_size)

    X_train, X_test, y_train, y_test = train_test_split(
        obs, actions, test_size=0.33, random_state=42)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / (std + 1e-6)
    X_test = (X_test - mean) / (std + 1e-6)

    X_train = to_tensor(X_train, args.dtype)
    X_test = to_tensor(X_test, args.dtype)

    y_train = to_tensor(y_train, args.dtype)
    y_test = to_tensor(y_test, args.dtype)

    kwargs = set_args(args=args)

    train_dataset = RolloutsDataset(observations=X_train, actions=y_train,
                                    # transform=ToTensorWithType(),
                                    transform=None,
                                    )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.min_batch_size,
                              shuffle=True,
                              **kwargs)

    test_dataset = RolloutsDataset(observations=X_test, actions=y_test,
                                   # transform=ToTensorWithType(),
                                   transform=None,
                                   )

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.min_batch_size,
                             shuffle=False,
                             **kwargs)

    return train_loader, test_loader
