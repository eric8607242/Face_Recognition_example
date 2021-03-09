import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAveragePooling(nn.Module):
    def forward(self, x):
        return x.mean(3).mean(2)


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class ConvBNAct(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 activation,
                 bn_momentum,
                 bn_track_running_stats,
                 pad,
                 group=1,
                 *args,
                 **kwargs):

        super(ConvBNAct, self).__init__()

        assert activation in ["hswish", "relu", "prelu", "None", None]
        assert stride in [1, 2, 4]

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                pad,
                groups=group,
                bias=False))
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                out_channels,
                momentum=bn_momentum,
                track_running_stats=bn_track_running_stats))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())
        elif activation == "prelu":
            self.add_module("relu", nn.PReLU(out_channels))


class SEModule(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True)):
        super(SEModule, self).__init__()

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=in_channels // reduction,
                                      kernel_size=1,
                                      bias=True)
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(in_channels=in_channels // reduction,
                                     out_channels=in_channels,
                                     kernel_size=1,
                                     bias=True)
        self.excite_act = excite_act

    def forward(self, inputs):
        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        se_output = inputs * feature_excite_act

        return se_output


def channel_shuffle(x, groups=2):
    batch_size, c, w, h = x.shape
    group_c = c // groups
    x = x.view(batch_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, w, h)
    return x
