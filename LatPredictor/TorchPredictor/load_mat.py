# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import scipy.io


def clean(input_array):
    output_array = []
    for i in input_array:
        for j in i:
            if min(j.shape) > 0:
                output_array.append(j)
    return output_array


def load_mat(path_to_mat, net_name):
    mat = scipy.io.loadmat(path_to_mat)
    net = mat[net_name]
    net = list(net)

    IW = net[0][0][35]
    LW = net[0][0][36]
    b = net[0][0][37]

    IW = clean(IW)
    LW = clean(LW)
    b = clean(b)

    return IW, LW, b

if __name__=="__main__":
    IW, LW, b = load_mat('../matlab_net/matlab.mat','net')

    print(LW)
