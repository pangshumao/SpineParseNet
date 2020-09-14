# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit

    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_s, num_n,
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = num_s
        self.num_n = num_n


        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)

        # temp = self.conv_proj(x).to('cpu')
        # plt.subplot(421)
        # plt.imshow(temp[0, 0, 7, :, :])
        #
        # plt.subplot(422)
        # plt.imshow(temp[0, 1, 7, :, :])
        #
        # plt.subplot(423)
        # plt.imshow(temp[0, 2, 7, :, :])
        #
        # plt.subplot(424)
        # plt.imshow(temp[0, 3, 7, :, :])
        #
        # plt.subplot(425)
        # plt.imshow(temp[0, 4, 7, :, :])
        #
        # plt.subplot(426)
        # plt.imshow(temp[0, 5, 7, :, :])
        #
        # plt.subplot(427)
        # plt.imshow(temp[0, 6, 7, :, :])
        #
        # plt.subplot(428)
        # plt.imshow(temp[0, 7, 7, :, :])
        #
        # plt.show()

        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))

        return out


class GloRe_Unit_1D(GloRe_Unit):
    def __init__(self, num_in, num_s, num_n, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_1D, self).__init__(num_in, num_s, num_n,
                                            ConvNd=nn.Conv1d,
                                            BatchNormNd=nn.BatchNorm1d,
                                            normalize=normalize)

class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_s, num_n, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_2D, self).__init__(num_in, num_s, num_n,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)

class GloRe_Unit_3D(GloRe_Unit):
    def __init__(self, num_in, num_s, num_n, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_3D, self).__init__(num_in, num_s, num_n,
                                            ConvNd=nn.Conv3d,
                                            BatchNormNd=nn.BatchNorm3d,
                                            normalize=normalize)


if __name__ == '__main__':

    for normalize in [True, False]:
        data = torch.autograd.Variable(torch.randn(2, 32, 1))
        net = GloRe_Unit_1D(32, 16, normalize)
        print(net(data).size())

        data = torch.autograd.Variable(torch.randn(2, 32, 14, 14))
        net = GloRe_Unit_2D(32, 16, normalize)
        print(net(data).size())

        data = torch.autograd.Variable(torch.randn(2, 32, 8, 14, 14))
        net = GloRe_Unit_3D(32, 16, normalize)
        print(net(data).size())
