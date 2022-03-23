import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor


__all__ = ['googlenetlp']

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


def googlenetlp(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        pass


    cfg = [64, 64, 192, 64, 96, 128, 16, 32, 32, 128, 128, 192, 32, 96, 64, 192, 96, 208, 16, 48, 64, 160, 112, 224, 24,
           64, 64, 128, 128, 256, 24, 64, 64, 112, 144, 288, 32, 64, 64, 256, 160, 320, 32, 128, 128, 256, 160, 320, 32,
           128, 128, 384, 192, 384, 48, 128, 128]  # def

    # cfg = [round(item * 0.16) for item in cfg]
    # cfg = [48, 10, 175, 50, 19, 124, 6, 27, 21, 67, 44, 162, 30, 58, 1, 91, 81, 181, 3, 36, 58, 139, 90, 123, 3, 27, 56, 62,
    #  59, 137, 16, 36, 4, 64, 23, 29, 25, 63, 58, 149, 143, 75, 24, 23, 63, 180, 160, 214, 26, 59, 105, 269, 106, 289,
    #  35, 111, 50]


    model = GoogLeNet(cfg, **kwargs)

    return model


class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, cfg, num_classes=100, aux_logits=False, transform_input=False, init_weights=True,
                 blocks=None):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, cfg[0], kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(cfg[0], cfg[1], kernel_size=1)
        self.conv3 = conv_block(cfg[1], cfg[2], kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        cur = 2
        self.inception3a = inception_block(cfg[2], cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4], cfg[cur + 5],
                                           cfg[cur + 6])
        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        cur = cur + 6
        self.inception3b = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        cur = cur + 6
        self.inception4a = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        # aux1_input = in_channel
        cur = cur + 6
        # aux1_cfg = cfg[cur + 1]
        # cur = cur + 1

        self.inception4b = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        cur = cur + 6

        self.inception4c = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        cur = cur + 6
        self.inception4d = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        # aux2_input = in_channel
        cur = cur + 6
        # aux2_cfg = cfg[cur + 1]
        # cur = cur + 1

        self.inception4e = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        cur = cur + 6
        self.inception5a = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        cur = cur + 6
        self.inception5b = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])

        if aux_logits:
            # self.aux1 = inception_aux_block(aux1_input, num_classes, aux1_cfg)
            # self.aux2 = inception_aux_block(aux2_input, num_classes, aux2_cfg)
            pass
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        self.fc = nn.Linear(in_channel, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # type: (Tensor) -> Tuple[Tensor,Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)

        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14

        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)

        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1

        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # tmp = x
        # N x 1000 (num_classes)
        return x,  aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x, aux2, aux1):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> GoogLeNetOutputs
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x,  aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, cfg, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.conv = conv_block(in_channels, cfg, kernel_size=1)

        self.fc1 = nn.Linear(cfg * 16, 1024)

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
