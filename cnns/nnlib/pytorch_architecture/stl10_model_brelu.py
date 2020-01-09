import torch.nn as nn
from cnns.nnlib.pytorch_architecture import layer

Noise = layer.Noise
BReLU = layer.BReLU

model_urls = {
    'stl10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/stl10-866321e9.pth',
}

class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, noise_init):
    layers = []
    in_channels = 3
    noise_layer = Noise(noise_init)
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if i == 0:
                layers += [noise_layer, conv2d, nn.BatchNorm2d(out_channels, affine=False), BReLU()]
            else:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), BReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def stl10(n_channel, noise_init):
    cfg = [
        n_channel, 'M',
        2*n_channel, 'M',
        4*n_channel, 'M',
        4*n_channel, 'M',
        (8*n_channel, 0), (8*n_channel, 0), 'M'
    ]
    layers = make_layers(cfg, noise_init)
    model = SVHN(layers, n_channel=8*n_channel, num_classes=10)
    return model


