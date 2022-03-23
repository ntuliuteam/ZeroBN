import numpy as np
import torch
import torch.nn as nn


class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """

    def __init__(self, num_channels, select_num=None):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))
        self.num_channels = num_channels
        self.select_num = select_num

    def get_num_channels(self):
        return self.num_channels

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        if self.select_num:
            selected_index = np.squeeze(np.argwhere(np.ones(self.select_num)))
            if selected_index.size == 1:
                selected_index = np.resize(selected_index, (1,))

        else:
            selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
            # print('xxxxxxxxxxxxxxxxxxxxxxxx')
            # print(selected_index)
            # print('************************')
            if selected_index.size == 1:
                selected_index = np.resize(selected_index, (1,))

        # print(selected_index)

        output = input_tensor[:, selected_index, :, :]
        return output
