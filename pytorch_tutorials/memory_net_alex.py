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
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # calculate the size of the output from convolution part
        # before moving to the first fully connected layer
        def max_pool_cal(size, kernel=3, stride=2):
            return (size - kernel) // stride + 1

        def conv_cal(size, kernel=3, stride=1, padding=1):
            return 1 + (size + 2 * padding - kernel) // stride

        size = conv_cal(input_size, 11, 4, 2)
        size = max_pool_cal(size)
        size = conv_cal(size, 5, 1, 2)
        size = max_pool_cal(size)
        size = conv_cal(size)
        size = conv_cal(size)
        size = conv_cal(size)
        size = max_pool_cal(size)
        # print("size: ", size)
        self.img_size_after_features = size

        self.classifier = nn.Sequential(
            nn.Dropout(),  # a regularization method that implicitly creates an ensemble of neural networks
            nn.Linear(256 * size * size, 4096),  # maps from 256 * size * size to 4096 dimensional vector
            nn.ReLU(inplace=True),  # the standard non-linearity
            nn.Dropout(),
            nn.Linear(4096, 4096),  # maps from 4096 to another 4096 vector
            nn.ReLU(inplace=True),  # the standard non-linearity
            nn.Linear(4096, num_classes),
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

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model