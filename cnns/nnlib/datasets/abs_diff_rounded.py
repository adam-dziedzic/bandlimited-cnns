from cnns.nnlib.datasets.transformations.denormalize import Denormalize
from cnns.nnlib.datasets.cifar import cifar_std, cifar_mean
from torchvision.transforms import Normalize
from cnns.nnlib.datasets.transformations.rounding import \
    RoundingTransformation
import torch
from cnns.nnlib.datasets.cifar10_example import cifar10_example

def run():
    print("values per channel, diff between input and rounded image")
    for values_per_channel in [2 ** x for x in range(1, 12)]:
        rounder = RoundingTransformation(values_per_channel=values_per_channel)
        example = torch.tensor(cifar10_example)
        # print("input example: ", example)
        a = example.clone()
        a = Denormalize(std_array=cifar_std, mean_array=cifar_mean)(a)
        # print("denormalized a min: ", a.min())
        # print("denormalized a max: ", a.max())
        a = rounder(a)
        a = Normalize(std=cifar_std, mean=cifar_mean)(a)
        # print("a: ", a)
        # print("example: ", example)
        print(values_per_channel, ",", torch.sum(torch.abs(a - example)).item())


if __name__ == "__main__":
    run()