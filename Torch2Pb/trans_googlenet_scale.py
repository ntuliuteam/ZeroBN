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


class MaxPool2d:
    def __init__(self, kernel_size=3, stride=1, padding=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if self.padding == 'SAME' or self.padding == 'VALID':
            x = tf.nn.max_pool2d(x, ksize=[1, self.kernel_size, self.kernel_size, 1],
                                 strides=[1, self.stride, self.stride, 1],
                                 padding=self.padding)
        else:
            paddings = tf.constant(self.padding)
            x = tf.pad(x, paddings, "CONSTANT")
            x = tf.nn.max_pool2d(x, ksize=[1, self.kernel_size, self.kernel_size, 1],
                                 strides=[1, self.stride, self.stride, 1],
                                 padding='VALID')
        return x


class BasicConv2d:
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = 'SAME'
        else:
            self.padding = padding

    def forward(self, x):
        global conv_weight_trans
        global bn_weight_trnas
        w_conv = tf.Variable(tf.constant(Conv_weight[conv_weight_trans]))
        print(Conv_weight[conv_weight_trans].shape)
        x = tf.nn.conv2d(x, w_conv, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        conv_weight_trans = conv_weight_trans + 1
        mean = tf.Variable(tf.constant(BN_mean[bn_weight_trnas]))
        var = tf.Variable(tf.constant(BN_var[bn_weight_trnas]))
        scale = tf.Variable(tf.constant(BN_weight[bn_weight_trnas]))
        shift = tf.Variable(tf.constant(BN_bias[bn_weight_trnas]))
        epsilon = 0.001
        bn_weight_trnas = bn_weight_trnas + 1
        x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

        x = tf.nn.relu(x)
        return x


class Inception:
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, conv_block=None):
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2_1 = conv_block(in_channels, ch3x3red, kernel_size=1)
        self.branch2_2 = conv_block(ch3x3red, ch3x3, kernel_size=3)
        self.branch3_1 = conv_block(in_channels, ch5x5red, kernel_size=1)
        self.branch3_2 = conv_block(ch5x5red, ch5x5, kernel_size=3)
        self.branch4_1 = MaxPool2d(kernel_size=3, stride=1, padding=[[0, 0, ], [1, 1], [1, 1], [0, 0]])
        self.branch4_2 = conv_block(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1.forward(x)
        branch2 = self.branch2_1.forward(x)
        branch2 = self.branch2_2.forward(branch2)
        branch3 = self.branch3_1.forward(x)
        branch3 = self.branch3_2.forward(branch3)
        branch4 = self.branch4_1.forward(x)
        branch4 = self.branch4_2.forward(branch4)
        outputs = [branch1, branch2, branch3, branch4]

        x = tf.concat(outputs, 3)

        return x


class googlenet:
    def __init__(self, cfg, num_classes=100, blocks=None):
        if blocks is None:
            blocks = [BasicConv2d, Inception]
        conv_block = blocks[0]
        inception_block = blocks[1]

        self.num_classes = num_classes

        self.conv1 = conv_block(3, cfg[0], kernel_size=7, strides=2, padding=[[0, 0, ], [3, 2], [3, 2], [0, 0]])
        self.maxpool1 = MaxPool2d(kernel_size=3, stride=2, padding=[[0, 0, ], [0, 2], [0, 2], [0, 0]])
        self.conv2 = conv_block(cfg[0], cfg[1], kernel_size=1)
        self.conv3 = conv_block(cfg[1], cfg[2], kernel_size=3)
        self.maxpool2 = MaxPool2d(kernel_size=3, stride=2, padding=[[0, 0, ], [0, 2], [0, 2], [0, 0]])

        cur = 2
        self.inception3a = inception_block(cfg[2], cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4], cfg[cur + 5],
                                           cfg[cur + 6])
        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        cur = cur + 6
        self.inception3b = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])
        self.maxpool3 = MaxPool2d(kernel_size=3, stride=2, padding=[[0, 0, ], [0, 2], [0, 2], [0, 0]])

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]
        cur = cur + 6
        self.inception4a = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])

        in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]

        cur = cur + 6

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

        cur = cur + 6

        self.inception4e = inception_block(in_channel, cfg[cur + 1], cfg[cur + 2], cfg[cur + 3], cfg[cur + 4],
                                           cfg[cur + 5],
                                           cfg[cur + 6])
        self.maxpool4 = MaxPool2d(kernel_size=2, stride=2, padding=[[0, 0, ], [0, 1], [0, 1], [0, 0]])

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

        self.fc_in_channel = cfg[cur + 1] + cfg[cur + 3] + cfg[cur + 5] + cfg[cur + 6]

    def forward(self, x):
        x = self.conv1.forward(x)
        # N x 64 x 112 x 112
        x = self.maxpool1.forward(x)
        # N x 64 x 56 x 56
        x = self.conv2.forward(x)
        # N x 64 x 56 x 56
        x = self.conv3.forward(x)
        # N x 192 x 56 x 56
        x = self.maxpool2.forward(x)
        # N x 192 x 28 x 28
        x = self.inception3a.forward(x)
        # N x 256 x 28 x 28
        x = self.inception3b.forward(x)
        # N x 480 x 28 x 28
        x = self.maxpool3.forward(x)
        # N x 480 x 14 x 14

        x = self.inception4a.forward(x)

        x = self.inception4b.forward(x)
        # N x 512 x 14 x 14
        x = self.inception4c.forward(x)
        # N x 512 x 14 x 14
        x = self.inception4d.forward(x)
        x = self.inception4e.forward(x)
        # N x 832 x 14 x 14
        x = self.maxpool4.forward(x)

        # N x 832 x 7 x 7
        x = self.inception5a.forward(x)
        # N x 832 x 7 x 7
        x = self.inception5b.forward(x)

        x = tf.nn.avg_pool2d(x, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')

        in_size = len(last_nonzero)
        x = tf.reshape(x, [-1, in_size])

        w_fc = tf.Variable(tf.constant(Linear_weight[0]))
        b_fc = tf.Variable(tf.constant((Linear_bias[0])))
        x = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc, name="Output")

        return x


def main():
    sess = tf.InteractiveSession()

    x_input = tf.placeholder("float", [None, 224, 224, 3], name="Input")

    cfg = [64, 64, 192, 64, 96, 128, 16, 32, 32, 128, 128, 192, 32, 96, 64, 192, 96, 208, 16, 48, 64, 160, 112, 224, 24,
           64, 64, 128, 128, 256, 24, 64, 64, 112, 144, 288, 32, 64, 64, 256, 160, 320, 32, 128, 128, 256, 160, 320, 32,
           128, 128, 384, 192, 384, 48, 128, 128]  # def

    cfg = [round(i * 2) for i in cfg]

    model = googlenet(cfg=cfg)
    out = model.forward(x_input)

    sess.run(tf.global_variables_initializer())

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Input', 'Output'])
    with tf.gfile.FastGFile(save, mode='wb') as f:
        f.write(constant_graph.SerializeToString())


if __name__ == '__main__':

    conv_weight_trans = 0
    bn_weight_trnas = 0

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

    model = models.__dict__['googlenet_scale']()
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
    last_nonzero_tmp_1 = None
    last_nonzero_tmp_2 = None

    last_nonzero_four = [None, None, None, None]

    ll = 0
    incep = 0

    for m in model.modules():

        if isinstance(m, nn.Linear):
            (one, len_one) = last_nonzero_four[0]
            (two, len_two) = last_nonzero_four[1]
            (three, len_three) = last_nonzero_four[2]
            (four, len_four) = last_nonzero_four[3]
            last_nonzero = np.union1d(one, two.__add__(len_one))
            last_nonzero = np.union1d(last_nonzero, three.__add__(len_one + len_two))
            last_nonzero = np.union1d(last_nonzero, four.__add__(len_one + len_two + len_three))

            # last_nonzero = nonzero_index
            last_nonzero_four = [None, None, None, None]
            process = m.weight.data.cpu().transpose(0, 1).numpy()
            process = process[last_nonzero]
            Linear_weight.append(process)
            Linear_bias.append(m.bias.data.cpu().numpy())

        if isinstance(m, nn.Conv2d):
            Conv_weight.append(m.weight.data.cpu().numpy())

        if isinstance(m, nn.BatchNorm2d):
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
            weight_len = len(weight)

            if ll < 3:

                if last_nonzero is not None:
                    process = Conv_weight[conv_last][nonzero_index]
                    process = torch.tensor(process)
                    process = process.transpose(0, 1).numpy()
                    process = process[last_nonzero]
                    process = torch.tensor(process).transpose(1, 0).transpose(0, 3).transpose(1, 2).transpose(0,
                                                                                                              1).numpy()
                    Conv_weight[conv_last] = process

                else:
                    process = Conv_weight[conv_last][nonzero_index]
                    process = torch.tensor(process)
                    process = process.transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                    Conv_weight[conv_last] = process

                last_layer = len(nonzero_index)
                last_nonzero = nonzero_index
            else:
                if None not in last_nonzero_four:
                    if incep == 0:
                        (one, len_one) = last_nonzero_four[0]
                        (two, len_two) = last_nonzero_four[1]
                        (three, len_three) = last_nonzero_four[2]
                        (four, len_four) = last_nonzero_four[3]
                        last_nonzero = np.union1d(one, two.__add__(len_one))
                        last_nonzero = np.union1d(last_nonzero, three.__add__(len_one + len_two))
                        last_nonzero = np.union1d(last_nonzero, four.__add__(len_one + len_two + len_three))

                        # last_nonzero = nonzero_index
                        last_nonzero_four = [None, None, None, None]

                if incep == 0 or incep == 1 or incep == 3 or incep == 5:
                    process = Conv_weight[conv_last][nonzero_index]
                    process = torch.tensor(process)
                    process = process.transpose(0, 1).numpy()
                    process = process[last_nonzero]
                    process = torch.tensor(process).transpose(1, 0).transpose(0, 3).transpose(1, 2).transpose(0,
                                                                                                              1).numpy()
                    Conv_weight[conv_last] = process
                    if incep == 0:
                        last_nonzero_four[0] = (nonzero_index, weight_len)
                    if incep == 5:
                        last_nonzero_four[3] = (nonzero_index, weight_len)
                    if incep == 1:
                        last_nonzero_tmp_1 = nonzero_index
                    if incep == 3:
                        last_nonzero_tmp_2 = nonzero_index
                else:
                    if incep == 2:
                        process = Conv_weight[conv_last][nonzero_index]
                        process = torch.tensor(process)
                        process = process.transpose(0, 1).numpy()
                        process = process[last_nonzero_tmp_1]
                        process = torch.tensor(process).transpose(1, 0).transpose(0, 3).transpose(1, 2).transpose(0,
                                                                                                                  1).numpy()
                        Conv_weight[conv_last] = process

                        last_nonzero_four[1] = (nonzero_index, weight_len)
                    if incep == 4:
                        process = Conv_weight[conv_last][nonzero_index]
                        process = torch.tensor(process)
                        process = process.transpose(0, 1).numpy()
                        process = process[last_nonzero_tmp_2]
                        process = torch.tensor(process).transpose(1, 0).transpose(0, 3).transpose(1, 2).transpose(0,
                                                                                                                  1).numpy()
                        Conv_weight[conv_last] = process

                        last_nonzero_four[2] = (nonzero_index, weight_len)

                incep = incep + 1
                if incep > 5:
                    incep = 0


            ll = ll + 1
    main()
