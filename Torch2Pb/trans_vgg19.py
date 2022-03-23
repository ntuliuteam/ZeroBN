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


def avg_pool1x1(x):
    return tf.nn.avg_pool2d(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')


def max_pool2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def classifier(x, num_classes):
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    in_size = last_layer * 7 * 7
    x = tf.reshape(x, [-1, in_size])
    # w_fc = weight_variable([in_size, 4096])
    w_fc = tf.Variable(tf.constant(Linear_weight[0]))
    b_fc = tf.Variable(tf.constant((Linear_bias[0])))
    x = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc)
    x = tf.nn.relu(x)

    w_fc = tf.Variable(tf.constant(Linear_weight[1]))
    b_fc = tf.Variable(tf.constant((Linear_bias[1])))
    x = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc)
    x = tf.nn.relu(x)

    w_fc = tf.Variable(tf.constant(Linear_weight[2]))
    b_fc = tf.Variable(tf.constant((Linear_bias[2])))
    out = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc, name='Output')

    return out


class VGG:
    def __init__(self, feature, num_classes=100):
        self.feature = feature
        self.avgpool = avg_pool1x1
        self.classifier = classifier
        self.num_classes = num_classes

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = classifier(x, self.num_classes)
        return x


def make_layers(cfg, x, batch_norm=False):
    # in_channels = 3
    cur = 0
    for v in cfg:
        if v == 'M':
            x = max_pool2x2(x)
        else:
            if batch_norm:
                w_conv = tf.Variable(tf.constant(Conv_weight[cur]))
                b_conv = tf.Variable(tf.constant(Conv_bias[cur]))
                x = tf.nn.bias_add(conv2d(x, w_conv), b_conv)

                mean = tf.Variable(tf.constant(BN_mean[cur]))
                var = tf.Variable(tf.constant(BN_var[cur]))
                scale = tf.Variable(tf.constant(BN_weight[cur]))
                shift = tf.Variable(tf.constant(BN_bias[cur]))
                epsilon = 0.00001
                x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

                x = tf.nn.relu(x)
            else:
                w_conv = tf.Variable(tf.constant(Conv_weight[cur]))
                b_conv = tf.Variable(tf.constant(Conv_bias[cur]))
                x = tf.nn.bias_add(conv2d(x, w_conv), b_conv)

                x = tf.nn.relu(x)
            # in_channels = v
            cur = cur + 1
    return x


def feature_func(x):
    # NOTE: the number in cfgs will not be used. We remove channels according to the zeros of BN
    cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return make_layers(cfgs, x, True)


def vgg19():
    return VGG(feature_func, 100)


def main():
    sess = tf.InteractiveSession()

    x_input = tf.placeholder("float", [None, 224, 224, 3], name="Input")
    vgg = vgg19()
    x_tmp = vgg.forward(x_input)

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

    model = models.__dict__['vgg19_bn']()
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    checkpoint = torch.load(torch_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    Conv_weight = []
    Conv_bias = []
    BN_weight = []
    BN_bias = []
    BN_mean = []
    BN_var = []
    Linear_weight = []
    Linear_bias = []

    last_layer = 0
    last_nonzero = None

    ll = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if ll == 0:
                process = m.weight.data.cpu().numpy().reshape(4096, 512, 7, 7)
                process = torch.tensor(process).transpose(0, 1).numpy()
                process = process[last_nonzero]
                process = torch.tensor(process).transpose(0, 1).numpy()
                process = process.reshape(4096, last_layer * 7 * 7)
                process = torch.tensor(process).transpose(0, 1).numpy()

                Linear_weight.append(process)
                Linear_bias.append(m.bias.data.cpu().numpy())

                ll = ll + 1
            else:
                Linear_weight.append(m.weight.data.transpose(0, 1).cpu().numpy())
                Linear_bias.append(m.bias.data.cpu().numpy())

        if isinstance(m, nn.Conv2d):

            Conv_weight.append(m.weight.data.cpu().numpy())
            Conv_bias.append(m.bias.data.cpu().numpy())

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

            if last_nonzero is not None:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 1).numpy()
                process = process[last_nonzero]
                process = torch.tensor(process).transpose(1, 0).transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                Conv_weight[conv_last] = process

                process = Conv_bias[conv_last][nonzero_index]
                Conv_bias[conv_last] = process
            else:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                Conv_weight[conv_last] = process

                process = Conv_bias[conv_last][nonzero_index]
                Conv_bias[conv_last] = process

            last_layer = len(nonzero_index)
            last_nonzero = nonzero_index

    main()
