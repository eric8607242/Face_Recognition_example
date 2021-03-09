import os.path as osp
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel

class MobileFaceNet(BaseModel):
    CONFIG_PATH = osp.join(osp.dirname(osp.abspath(__file__)), 'config.yml')
    def __init__(self, bn_momentum=0.1, bn_track_running_stats=True, config_path=None):
        super(MobileFaceNet, self).__init__()

        config_path = self.CONFIG_PATH if config_path is None else config_path
        self.config = self._parse_config(config_path)

        self.model = self._construct_model(self.config["model_config"], bn_momentum, bn_track_running_stats)

        self._initialize_weights()

    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)

        return x



