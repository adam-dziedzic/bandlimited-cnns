import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, input_size=256):
        super(AlexNet, self).__init__()
        self.input_channel = 3
        self.img_size_to_features = input_size  # the size of the image to the first layer is the size of the input
        size = input_size
        self.img_size_after_features = size
        # print("size: ", size)

        self.classifier = nn.Sequential(
            nn.Linear(3 * size * size, num_classes),
        )

    def forward(self, x):
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