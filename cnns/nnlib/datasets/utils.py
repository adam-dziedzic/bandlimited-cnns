from __future__ import print_function
import torch.utils.data as data
import torch


class SingleImage(data.Dataset):
    def __init__(self, image, label, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = False  # training set or test set

        self.test_data = [image]
        self.test_labels = [label]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index > 0:
            raise Exception('This is a single image dataset.')
        img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.test_data)


def get_loader(image, label, num_workers=1, pin_memory=True, batch_size=1):
    test_dataset = SingleImage(image=image, label=label)
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              **kwargs)
    return test_loader
