from __future__ import absolute_import
import math
import torch.nn as nn
from .channel_selection import channel_selection

__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # print(inplanes, planes, cfg)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes, cfg[0])
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # print('tt\t',x.shape)
        out = self.bn1(x)
        out = self.select(out)
        # print(self.select.get_num_channels())
        # print('ff\t',out.shape)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class resnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16] * (n - 1), [64, 32, 32], [128, 32, 32] * (n - 1), [128, 64, 64],
                   [256, 64, 64] * (n - 1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

            # cfg = [round(i*0.5) for i in cfg]


            # cfg = [11, 16, 14, 3, 6, 8, 5, 7, 9, 2, 2, 3, 3, 6, 1, 4, 6, 4, 7, 8, 9, 4, 8, 7, 6, 10, 9, 6, 7, 8, 7, 7,
            #        11, 1, 4, 11, 4, 8, 14, 6, 8, 14, 8, 5, 14, 3, 3, 7, 2, 7, 15, 2, 5, 11, 13, 29, 32, 11, 9, 28, 13,
            #        19, 31, 12, 13, 29, 4, 8, 28, 4, 15, 28, 5, 15, 25, 13, 14, 32, 6, 8, 29, 9, 17, 27, 9, 16, 28, 10,
            #        16, 26, 15, 18, 28, 12, 13, 25, 7, 10, 16, 9, 18, 27, 7, 6, 19, 12, 14, 19, 99, 63, 64, 7, 18, 61, 4,
            #        21, 60, 11, 36, 59, 13, 37, 62, 8, 32, 63, 16, 46, 62, 25, 55, 62, 26, 53, 60, 18, 38, 58, 29, 39,
            #        57, 38, 49, 54, 44, 47, 51, 32, 37, 56, 38, 37, 51, 33, 33, 53, 47, 42, 49, 41, 34, 47, 76]


        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:3 * n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[3 * n:6 * n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[6 * n:9 * n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion, cfg[-1])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3 * i: 3 * (i + 1)]))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
