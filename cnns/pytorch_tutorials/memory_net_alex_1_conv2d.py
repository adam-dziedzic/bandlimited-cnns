import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, input_size=256):
        self.input_channel = 3
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # calculate the size of the output from convolution part before moving to the first fully connected layer
        def max_pool(size, kernel=3, stride=2):
            return (size - kernel) // stride + 1

        def conv(size, kernel=3, stride=1, padding=1):
            return 1 + (size + 2 * padding - kernel) // stride

        size = conv(input_size, 11, 4, 2)
        size = max_pool(size)
        print("size: ", size)

        self.classifier = nn.Sequential(
            nn.Linear(64 * size * size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print("x shape after features: ", x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        print("No possibility of pretraining for a small network.")
        import sys
        sys.exit(1)
    model = AlexNet(**kwargs)
    return model