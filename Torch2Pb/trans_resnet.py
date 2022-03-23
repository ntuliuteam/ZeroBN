# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import models
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.python.framework import graph_util


class Conv3x3:
    def __init__(self, in_planes, out_planes, stride=1):
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

    def forward(self, x):
        global cur_conv
        w_conv = tf.Variable(tf.constant(Conv_weight[cur_conv]))
        cur_conv = cur_conv + 1

        return tf.nn.conv2d(x, w_conv, strides=[1, self.stride, self.stride, 1], padding=[[0, 0], [1,
                                                                                                   1], [1, 1], [0, 0]])


class Conv1x1:
    def __init__(self, in_planes, out_planes, stride=1):
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

    def forward(self, x):
        global cur_conv
        w_conv = tf.Variable(tf.constant(Conv_weight[cur_conv]))
        cur_conv = cur_conv + 1
        return tf.nn.conv2d(x, w_conv, strides=[1, self.stride, self.stride, 1], padding='SAME')


class Conv2D:
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=7, padding=None, bias=False):
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

    def forward(self, x):
        global cur_conv
        if self.bias:
            print('no bias')
            exit()
        else:
            w_conv = tf.Variable(tf.constant(Conv_weight[cur_conv]))
            x = tf.nn.conv2d(x, w_conv, strides=[1, self.stride, self.stride, 1], padding=self.padding)
            cur_conv = cur_conv + 1
            return x


class BN:
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        size = self.size
        global cur_bn
        mean = tf.Variable(tf.constant(BN_mean[cur_bn]))
        var = tf.Variable(tf.constant(BN_var[cur_bn]))
        scale = tf.Variable(tf.constant(BN_weight[cur_bn]))
        shift = tf.Variable(tf.constant(BN_bias[cur_bn]))
        epsilon = 1e-5
        cur_bn = cur_bn + 1
        x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)
        return x


class Sequential:
    def __init__(self, *args):
        self.args = args

    def forward(self, x):
        output = x
        for lay in self.args:
            output = lay.forward(output)
        return output


class Bottleneck:

    def __init__(self, cfg, stride=1, downsample=None):
        self.conv1 = Conv1x1(cfg[0], cfg[1])
        self.bn1 = BN(cfg[1])
        self.conv2 = Conv3x3(cfg[1], cfg[2], stride)
        self.bn2 = BN(cfg[2])
        self.conv3 = Conv1x1(cfg[2], cfg[3])
        self.bn3 = BN(cfg[3])

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = tf.nn.relu(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = tf.nn.relu(out)

        out = self.conv3.forward(out)
        out = self.bn3.forward(out)

        if self.downsample is not None:
            identity = self.downsample.forward(x)

        out = out + identity
        out = tf.nn.relu(out)

        return out


class MakeLayer:
    def __init__(self, block, blocks, cfg, stride=1):

        downsample = Sequential(Conv1x1(cfg[0], cfg[-1], stride), BN(cfg[-1]))

        layers = [block(cfg[0:4], stride, downsample)]

        for i in range(1, blocks):
            layers.append(block(cfg[3 * i:3 * (i + 1) + 1]))

        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


class ResNet:
    def __init__(self, block, layers, cfg, num_classes=100):
        self.inplanes = 64
        self.dilation = 1

        self.cfg = cfg
        self.num_classes = num_classes

        self.conv1 = Conv2D(3, cfg[0], kernel_size=7, stride=2, bias=False, padding=[[0, 0], [3,
                                                                                              2], [3, 2], [0, 0]])
        self.bn1 = BN(cfg[0])

        cur = 0

        self.layer1 = MakeLayer(block, layers[0], cfg[cur:cur + layers[0] * 3 + 1])
        cur = cur + layers[0] * 3

        self.layer2 = MakeLayer(block, layers[1], cfg[cur:cur + layers[1] * 3 + 1], stride=2)
        cur = cur + layers[1] * 3

        self.layer3 = MakeLayer(block, layers[2], cfg[cur:cur + layers[2] * 3 + 1], stride=2)
        cur = cur + layers[2] * 3

        self.layer4 = MakeLayer(block, layers[3], cfg[cur:cur + layers[3] * 3 + 1], stride=2)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = tf.nn.relu(x)
        paddings = tf.constant([[0, 0, ], [1, 0], [1, 0], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT")
        x = tf.nn.max_pool2d(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.layer4.forward(x)

        x = tf.nn.avg_pool2d(x, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')

        in_size = self.cfg[-1]
        x = tf.reshape(x, [-1, in_size])

        w_fc = tf.Variable(tf.constant(Linear_weight[0]))
        b_fc = tf.Variable(tf.constant((Linear_bias[0])))
        x = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc, name="Output")

        return x


def resnet50():
    cfg_def = [[64], [64, 64, 256] * 3, [128, 128, 512] * 4, [256, 256, 1024] * 6, [512, 512, 2048] * 3]
    cfg_def = [item for sub_list in cfg_def for item in sub_list]

    cfg = [[64], [64, 64, 256] * 3, [128, 128, 512] * 4, [256, 256, 1024] * 6, [512, 512, 2048] * 3]
    cfg = [item for sub_list in cfg for item in sub_list]
    for tmp in range(3, 49, 3):
        cfg[tmp] = cfg_def[tmp]

    return ResNet(Bottleneck, [3, 4, 6, 3], cfg)


def main():
    sess = tf.InteractiveSession()

    x_input = tf.placeholder("float", [None, 224, 224, 3], name="Input")

    model = resnet50()
    out = model.forward(x_input)
    sess.run(tf.global_variables_initializer())

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Input', 'Output'])
    with tf.gfile.FastGFile(save, mode='wb') as f:
        f.write(constant_graph.SerializeToString())


if __name__ == '__main__':

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    print(model_names)
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_model", help="The path to saved torch model checkpoint.", required=True)
    parser.add_argument("--save", help="The path where the frozen pb will be saved.", required=True)

    args = parser.parse_args()

    torch_model = args.torch_model
    save = args.save
    model = models.__dict__['resnet50_new']()
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(torch_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    Conv_weight = []
    # Conv_bias = []
    BN_weight = []
    BN_bias = []
    BN_mean = []
    BN_var = []
    Linear_weight = []
    Linear_bias = []

    last_layer = 0
    last_nonzero = None

    cur_bn = 0
    cur_conv = 0
    skip_list = [4, 5, 8, 11, 14, 15, 18, 21, 24, 27, 28, 31, 34, 37, 40, 43, 46, 47, 50, 53]  # = id + 1

    skip = 0
    first_nonezero = None

    for m in model.modules():

        if isinstance(m, nn.Linear):
            Linear_weight.append(m.weight.data.cpu().transpose(0, 1).numpy())
            Linear_bias.append(m.bias.data.cpu().numpy())

        if isinstance(m, nn.Conv2d):
            Conv_weight.append(m.weight.data.cpu().numpy())

        if isinstance(m, nn.BatchNorm2d):
            skip = skip + 1
            if skip == 5:
                last_nonzero = first_nonezero

            weight = m.weight.data.cpu().numpy()
            bias = m.bias.data.cpu().numpy()
            nonzero_index = np.union1d(np.nonzero(weight), np.nonzero(bias))
            if len(nonzero_index) == 0:
                nonzero_index = np.array([0])
            BN_weight.append(weight[nonzero_index])
            BN_bias.append(bias[nonzero_index])

            BN_mean.append(m.running_mean.cpu().numpy()[nonzero_index])
            BN_var.append(m.running_var.cpu().numpy()[nonzero_index])

            conv_last = len(BN_weight) - 1

            if last_nonzero is not None:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 1).numpy()
                process = process[last_nonzero]
                process = torch.tensor(process).transpose(1, 0).transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                Conv_weight[conv_last] = process

            else:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                Conv_weight[conv_last] = process

            last_layer = len(nonzero_index)
            last_nonzero = nonzero_index

            if skip in skip_list:
                last_nonzero = None
            if skip == 1:
                first_nonezero = nonzero_index

    main()
