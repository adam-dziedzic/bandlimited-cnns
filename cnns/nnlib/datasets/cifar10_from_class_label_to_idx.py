from cnns.nnlib.datasets.cifar10_from_class_idx_to_label import cifar10_from_class_idx_to_label

cifar10_from_class_label_to_idx = dict()

for key, value in cifar10_from_class_idx_to_label.items():
    cifar10_from_class_label_to_idx[value] = key


if __name__ == "__main__":
    print(cifar10_from_class_label_to_idx)
