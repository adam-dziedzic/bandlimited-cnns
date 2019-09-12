from __future__ import print_function
import os
import os.path
import pickle
import torch.utils.data as data
import torch
from cnns.nnlib.utils.general_utils import MemoryType


class Pickled(data.Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        self.root = os.path.expanduser(file)
        self.transform = transform
        self.target_transform = target_transform
        self.train = False  # training set or test set

        fo = open(file, 'rb')
        entry = pickle.load(fo)
        self.test_data = entry['images']
        self.test_labels = entry['labels']
        fo.close()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.test_data)


def get_pickled(file, sample_count=0, num_workers=4, pin_memory=True,
                batch_size=32):
    test_dataset = Pickled(file)
    if sample_count > 0:
        try:
            test_dataset.data = test_dataset.data[:sample_count]
            test_dataset.targets = test_dataset.targets[:sample_count]
        except AttributeError:
            test_dataset.test_data = test_dataset.test_data[:sample_count]
            test_dataset.test_labels = test_dataset.test_labels[:sample_count]
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              **kwargs)
    return test_loader, test_dataset


def get_pickled_args(file, args):
    pin_memory = False
    if args.memory_type is MemoryType.PINNED:
        pin_memory = True
    return get_pickled(file, sample_count=args.sample_count_limit,
                       num_workers=args.workers, pin_memory=pin_memory,
                       batch_size=args.test_batch_size)
