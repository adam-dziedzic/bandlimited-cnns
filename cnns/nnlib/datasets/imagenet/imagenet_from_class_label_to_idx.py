from cnns.nnlib.datasets.imagenet.imagenet_from_class_idx_to_label import \
    imagenet_from_class_idx_to_label

imagenet_from_class_label_to_idx = dict()

for key, value in imagenet_from_class_idx_to_label.items():
    imagenet_from_class_label_to_idx[value] = key

if __name__ == "__main__":
    print(imagenet_from_class_label_to_idx)
