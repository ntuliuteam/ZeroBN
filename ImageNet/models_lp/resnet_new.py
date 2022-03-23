import torch
import torch.nn as nn
import numpy as np


__all__ = ['resnet50_newlp']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    # expansion = 4

    # input 4 cfg => conv1/2/3 i/o
    def __init__(self, cfg, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1


        # self.select = channel_selection(inplanes, cfg[0])
        self.conv1 = conv1x1(cfg[0], cfg[1])
        self.bn1 = norm_layer(cfg[1])

        self.conv2 = conv3x3(cfg[1], cfg[2], stride, groups, dilation)
        self.bn2 = norm_layer(cfg[2])

        self.conv3 = conv1x1(cfg[2], cfg[3])
        self.bn3 = norm_layer(cfg[3])

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group


        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        cur = 0
        self.layer1 = self._make_layer(block, layers[0], cfg[cur:cur + layers[0] * 3 + 1])
        cur = cur + layers[0] * 3
        self.layer2 = self._make_layer(block, layers[1], cfg[cur:cur + layers[1] * 3 + 1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        cur = cur + layers[1] * 3
        self.layer3 = self._make_layer(block, layers[2], cfg[cur:cur + layers[2] * 3 + 1], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        cur = cur + layers[2] * 3
        self.layer4 = self._make_layer(block, layers[3], cfg[cur:cur + layers[3] * 3 + 1], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.bn2 = nn.BatchNorm2d(cfg[-1])
        # self.select = channel_selection(512 * block.expansion, cfg[-1])

        self.fc = nn.Linear(cfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    # input 4 cfg
    def _make_layer(self, block, blocks, cfg, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        downsample = nn.Sequential(conv1x1(cfg[0], cfg[-1], stride),
                                   norm_layer(cfg[-1]))
        layers = []

        layers.append(block(cfg[0:4], stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        # self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(cfg[3 * i:3 * (i + 1) + 1], groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        x = self.conv1(x)


        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        # tmp = x
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, cfg, pretrained, progress, **kwargs):
    model = ResNet(block, layers, cfg, **kwargs)

    if pretrained:
        pass
    return model


def resnet50_newlp(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    cfg_def = [[64], [64, 64, 256] * 3, [128, 128, 512] * 4, [256, 256, 1024] * 6, [512, 512, 2048] * 3]
    cfg_def = [item for sub_list in cfg_def for item in sub_list]

    cfg = [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256,
           256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512,
           512, 512, 512, 512]

    for tmp in range(3, 49, 3):
        cfg.insert(tmp, cfg_def[tmp])

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], cfg, pretrained, progress,
                   **kwargs)
