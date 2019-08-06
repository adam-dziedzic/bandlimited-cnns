from torch.utils.data import Dataset
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from cnns.nnlib.utils.general_utils import MemoryType
from torch.utils.data import DataLoader
from cnns.nnlib.datasets.transformations.to_tensor import ToTensorWithType
import torch


class RolloutsDataset(Dataset):

    def __init__(self, observations, actions, transform=None):
        self.observations = observations
        self.actions = actions
        self.transform = transform

    def add_data(self, observations, actions):
        observations = np.array(observations)
        actions = np.array(actions).squeeze()

        observations = torch.from_numpy(observations)
        actions = torch.from_numpy(actions)

        observations = observations.to(self.observations.device).to(
            self.observations.dtype)
        actions = actions.to(self.actions.device).to(self.actions.dtype)

        self.observations = torch.cat((self.observations, observations), dim=0)
        self.actions = torch.cat((self.actions, actions), dim=0)

    def save_data(self, output_file, pickle_protocol=2):
        expert_data = {'observations': np.array(self.observations),
                       'actions': np.array(self.actions)}
        with open(output_file, 'wb') as file:
            pickle.dump(expert_data, file, pickle_protocol)

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
    print('filename: ', filename)
    with open(file=filename, mode="rb") as f:
        data = pickle.load(file=f)
    observations = data['observations']
    actions = np.squeeze(data['actions'])
    return observations, actions


def set_kwargs(args):
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


def read_data_and_shapes(args):
    obs, actions = read_data(args.rollout_file)

    args.input_size = obs.shape[1]
    args.output_size = actions.shape[1]

    print('input and output sizes: ', args.input_size, ' ', args.output_size)
    return obs, actions


def limit_data(args, obs, actions):
    sample_count = args.sample_count_limit
    if sample_count > 0:
        obs, actions = obs[:sample_count], actions[:sample_count]
    return obs, actions


def get_rollouts_dataset(args):
    obs, actions = read_data_and_shapes(args=args)

    obs, actions = limit_data(args=args, obs=obs, actions=actions)

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

    kwargs = set_kwargs(args=args)

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

    return train_loader, test_loader, train_dataset, test_dataset
