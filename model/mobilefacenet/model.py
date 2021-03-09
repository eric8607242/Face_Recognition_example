import os.path as osp
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..network_utils import ConvBNAct

class IRBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 activation,
                 expansion_rate,
                 bn_momentum,
                 bn_track_running_stats,
                 point_group):
        super(IRBlock, self).__init__()

        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_channel = int(in_channels * expansion_rate)

        if expansion_rate == 1:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNAct(
                in_channels=in_channels,
                out_channels=hidden_channel,
                kernel_size=1,
                stride=1,
                activation=activation,
                bn_momentum=bn_momentum,
                bn_track_running_stats=bn_track_running_stats,
                group=point_group,
                pad=0)
        self.depthwise = ConvBNAct(
            in_channels=hidden_channel,
            out_channels=hidden_channel,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            bn_momentum=bn_momentum,
            bn_track_running_stats=bn_track_running_stats,
            pad=(kernel_size // 2),
            group=hidden_channel)

        self.point_wise_1 = ConvBNAct(
            in_channels=hidden_channel,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            activation=None,
            bn_momentum=bn_momentum,
            bn_track_running_stats=bn_track_running_stats,
            group=point_group,
            pad=0)

    def forward(self, x):
        y = self.point_wise(x)
        y = self.depthwise(y)
        y = self.point_wise_1(y)

        y = y + x if self.use_res_connect else y

        return y

def get_block(block_type,
              in_channels,
              out_channels,
              kernel_size,
              stride,
              activation,
              group,
              bn_momentum,
              bn_track_running_stats,
              *args,
              **kwargs):

    if block_type == "Mobile":
        # Inverted Residual Block of MobileNet
        expansion_rate = kwargs["expansion_rate"]
        point_group = kwargs["point_group"] if "point_group" in kwargs else 1

        block = IRBlock(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        activation=activation,
                        expansion_rate=expansion_rate,
                        bn_momentum=bn_momentum,
                        bn_track_running_stats=bn_track_running_stats,
                        point_group=point_group)

    elif block_type == "global_average":
        block = GlobalAveragePooling()

    elif block_type == "Conv":
        block = ConvBNAct(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          activation=activation,
                          bn_momentum=bn_momentum,
                          bn_track_running_stats=bn_track_running_stats,
                          group=group,
                          pad=(kernel_size // 2))

    else:
        raise NotImplementedError

    return block


class MobileFaceNet(BaseModel):
    CONFIG_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'config.yml')
    def __init__(self, bn_momentum=0.1, bn_track_running_stats=True, config_path=None):
        super(MobileFaceNet, self).__init__()

        config_path = self.CONFIG_PATH if config_path is None else config_path
        self.config = self._parse_config(config_path)

        layers = []
        for l_cfg in self.config["model_config"]:
            block_type, in_channels, out_channels, stride, kernel_size, group, activation, kwargs = l_cfg

            layer = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              group=group,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **kwargs)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        self._initialize_weights()

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)

        return x



