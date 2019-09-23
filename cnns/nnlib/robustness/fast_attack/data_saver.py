import numpy as np
import pickle
from cnns.nnlib.utils.general_utils import get_log_time


class DataSaver:

    def __init__(self, dataset):
        self.adv_images = None
        self.adv_labels = None
        self.org_images = None
        self.org_labels = None
        self.dataset = dataset

    def add_data(self, adv_images, adv_labels, org_images, org_labels):
        if self.adv_images is None:
            self.adv_images = adv_images
        else:
            self.adv_images = np.append(self.adv_images, adv_images, axis=0)


        if self.adv_labels is None:
            self.adv_labels = adv_labels
        else:
            self.adv_labels = np.append(self.adv_labels, adv_labels, axis=0)


        if self.org_images is None:
            self.org_images = org_images
        else:
            self.org_images = np.append(self.org_images, org_images, axis=0)

        if self.org_labels is None:
            self.org_labels = org_labels
        else:
            self.org_labels = np.append(self.org_labels, org_labels, axis=0)

    def save_adv_org(self):
        save_data = []
        save_data.append(['adv', self.adv_images, self.adv_labels])
        save_data.append(['org', self.org_images, self.org_labels])
        for name, images, labels in save_data:
            self.save_file(name=name + '-' + self.dataset, images=images,
                      labels=labels)

    def save_file(self, name, images, labels):
        save_time = get_log_time() + '-len-' + str(len(labels))
        pickle_protocol = 2
        file_name = save_time + '-' + name + '-images'
        with open(file_name, 'wb') as f:
            data = {'images': images, 'labels': labels}
            pickle.dump(data, f, pickle_protocol)