import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

__all__ = ['vgg19_bnlp']


class expend_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(expend_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv2d(x)
        return x


class vgg_block(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super(vgg_block, self).__init__()
        self.first = first
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if not first:
            self.expend = expend_block(in_channels, out_channels)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = Parameter(torch.Tensor([1, 0]))
        # self.shortcut.requires_grad = False

    def forward(self, x):
        if self.first:
            x = self.conv2d(x)
            x = self.batchnorm(x)
            x = self.relu(x)

        else:
            z = self.conv2d(x)
            z = self.batchnorm(z)

            z = self.relu(z)
            y = self.expend(x)

            x = self.shortcut[0] * z + self.shortcut[1] * y

        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=100, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    first = True
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            vgg_b = vgg_block(in_channels, v, first)
            first = False
            if batch_norm:
                layers += [vgg_b]
            else:
                exit(0)
                # layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}



def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        print('No Pretrained!!!')

    return model


def vgg19_bnlp (pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
