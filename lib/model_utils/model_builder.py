import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .network_utils import get_block


class Model(nn.Module):
    def __init__(
            self,
            model_config,
            bn_momentum=0.1,
            bn_track_running_stats=True):
        super(Model, self).__init__()
        self.model_config = model_config

        # First Stage
        self.first_stages = nn.ModuleList()
        for l_cfg in model_config["first"]:
            block_type, in_channels, out_channels, stride, kernel_size, group, activation, se, kwargs = l_cfg

            layer = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              group=group,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **kwargs)
            self.first_stages.append(layer)

        # Stage
        self.stages = nn.ModuleList()
        for l_cfg in model_config["stage"]:
            block_type, in_channels, out_channels, stride, kernel_size, group, activation, se, kwargs = l_cfg

            layer = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              group=group,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **kwargs)
            self.stages.append(layer)

        # Last Stage
        self.last_stages = nn.ModuleList()
        for l_cfg in model_config["last"]:
            block_type, in_channels, out_channels, stride, kernel_size, group, activation, se, kwargs = l_cfg

            layer = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              group=group,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **kwargs)
            self.last_stages.append(layer)

        self._initialize_weights()

    def forward(self, x):
        for i, l in enumerate(self.first_stages):
            x = l(x)

        for i, l in enumerate(self.stages):
            x = l(x)

        for i, l in enumerate(self.last_stages):
            x = l(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
