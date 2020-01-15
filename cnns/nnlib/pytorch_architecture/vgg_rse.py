'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn
from cnns.nnlib.pytorch_architecture import layer

Noise = layer.Noise
NoisePassBackward = layer.NoisePassBackward

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, init_noise, inner_noise, noise_type='standard'):
        super(VGG, self).__init__()
        self.init_noise = init_noise
        self.inner_noise = inner_noise
        self.noise_type = noise_type
        self.classifier = nn.Linear(512, 10)
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if i == 1:
                    noise = self.init_noise
                else:
                    noise = self.inner_noise

                if self.noise_type == 'backward':
                    noise_layer = NoisePassBackward(noise)
                elif self.noise_type == 'standard':
                    noise_layer = Noise(noise)
                else:
                    raise Exception(f'Unknown noise type: {self.noise_type}')
                layers += [
                        noise_layer,
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
