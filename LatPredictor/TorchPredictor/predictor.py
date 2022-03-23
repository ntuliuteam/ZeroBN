# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import csv
import math
import time

import numpy
import torch


def tansig(n):
    return 2 / (1 + torch.exp(-2 * n)) - 1


def sim(input_data):
    data_num = input_data.shape[0]
    min_input = torch.tensor([1, 1, 1, 1]).repeat(data_num, 1).transpose(0, 1)  # [1 1 1 1] need to be replaced

    max_input = torch.tensor([1, 1, 1, 1]).repeat(data_num, 1).transpose(0, 1)  # [1 1 1 1] need to be replaced
    min_target = torch.tensor(1)  # 1 need to be replaced
    max_target = torch.tensor(1)  # 1 need to be replaced

    input_data = input_data.transpose(0, 1)

    IW = torch.empty(16, 4)  # Here you need to fill in the weights of the Matlab trained network

    LW1 = torch.empty(12, 16)  # Here you need to fill in the weights of the Matlab trained network
    LW2 = torch.empty(1, 12)  # Here you need to fill in the weights of the Matlab trained network

    b1 = torch.empty(16, 1)  # Here you need to fill in the weights of the Matlab trained network

    b2 = torch.empty(12, 1)  # Here you need to fill in the weights of the Matlab trained network
    # b2 = b2.reshape(b2.shape[0])

    b3 = torch.tensor(1)  # 1 need to be replaced

    # normalization
    input_nor = 1.0 * (input_data - min_input) / (max_input - min_input) * 2 - 1

    # input layer
    output = torch.mm(IW, input_nor)
    output = output + b1
    output = tansig(output)

    # hidden layer 1
    output = torch.mm(LW1, output)
    output = output + b2
    output = tansig(output)

    # hidden layer 2
    output = torch.mm(LW2, output)
    output = output + b3

    output = output.reshape(data_num)

    # output layer
    out = (output + 1) / 2 * (max_target - min_target) + min_target

    out = torch.pow(out, -8)

    return out


if __name__ == '__main__':

    start = time.time()

    input_size = [224, 224, 112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 14, 14, 14]
    vgg19_cfg = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
    in_channel = [3] + vgg19_cfg[:-1]

    input_size = torch.tensor(input_size)
    vgg19_cfg = torch.tensor(vgg19_cfg)
    in_channel = torch.tensor(in_channel)

    layer_num = len(vgg19_cfg)

    layer_index = []
    for lay in range(layer_num):
        tmp = [lay + 1] * vgg19_cfg[lay]
        layer_index.append(tmp)

    layer_index = [item for sub_list in layer_index for item in sub_list]

    csvFile = open("y_i.csv", "r")
    reader = csv.reader(csvFile)
    bn = []

    for item in reader:
        bn.append(float(item[0]))

    # in the training, you can get bn from the zerobn function and replace bn here

    layer_index = torch.tensor(layer_index)
    bn = torch.tensor(bn)

    y, i = torch.sort(bn)

    layer_index = layer_index.gather(dim=0, index=i)


    def find_ratio(layer_index, start, end, constraint):

        while end - start > 0.01:
            current_r = (start + end) / 2
            lat = predict(layer_index, current_r)
            if lat > constraint:
                start = current_r

            elif lat < constraint:
                end = current_r
            else:
                return current_r

        return (start + end) / 2


    def predict(layer_index, ratio):

        num = int(layer_index.shape[0] * ratio) + 1
        if num >= layer_index.shape[0]:
            num = layer_index.shape[0] - 1

        lay_tmp = layer_index[:num]
        ind, sub = torch.unique(lay_tmp, return_counts=True)

        vgg19_cfg_tmp = vgg19_cfg.clone()
        for i in range(ind.shape[0]):
            vgg19_cfg_tmp[ind[i] - 1] = vgg19_cfg[ind[i] - 1] - sub[i]

        in_channel_tmp = torch.cat((torch.tensor([3]), vgg19_cfg_tmp.split(vgg19_cfg_tmp.shape[0] - 1)[0]), 0)

        input_data = torch.tensor(
            [[input_size[i], in_channel_tmp[i], 3, vgg19_cfg_tmp[i]] for i in range(len(input_size))])
        sim_all = sim(input_data)

        return sim_all.sum()


    start = time.time()

    print(find_ratio(layer_index, 0, 1, 10))

    print(time.time() - start)