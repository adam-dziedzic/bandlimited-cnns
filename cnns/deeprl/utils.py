import numpy as np
import pickle


def get_mean_std(obs):
    mean = np.mean(obs, axis=0)
    std = np.std(obs, axis=0)
    return mean, std


def read_data(filename):
    with open(file=filename, mode="rb") as f:
        data = pickle.load(file=f)
    observations = data['observations']
    actions = np.squeeze(data['actions'])
    return observations, actions


