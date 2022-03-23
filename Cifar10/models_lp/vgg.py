# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

__all__ = ['vgg_lp']

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class expend_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(expend_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2d = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.conv2d(x)

            #it is also can be implemented to just skip by zero
            #b = x.shape
            #x = torch.zeros(b[0], self.out_channels, b[2], b[3]).cuda()

        return x


class vgg_block(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super(vgg_block, self).__init__()
        self.first = first
        # self.conv = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if not first:
            self.expend = expend_block(in_channels, out_channels)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = Parameter(torch.Tensor([1, 0]))
        self.shortcut.requires_grad = False

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


class vgg_lp(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(vgg_lp, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        # self.cfg = cfg
        self.feature = self.make_layers(cfg, True)

        # in_channels = 3

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        first = True
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                vgg_b = vgg_block(in_channels, v, first)
                first = False
                if batch_norm:
                    layers += [vgg_b]
                else:
                    # layers += [conv2d, nn.ReLU(inplace=True)]
                    exit(0)
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)

        x = nn.AvgPool2d(2)(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = vgg_lp()
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y = net(x)
    print(y.data.shape)
