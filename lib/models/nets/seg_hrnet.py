import torch
import torch.nn as nn
import torch.functional as F
from lib.models.backbones.basenet import BaseNet

import logging

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1

class HRNet(BaseNet):
    def __init__(self,config,last_input_channel = 720,**kwargs):
        super(HRNet,self).__init__(config,**kwargs)
        self.last_input_channel = last_input_channel

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.last_input_channel,
                out_channels=self.last_input_channel,
                kernel_size=1,
                stride=1,
                padding=0),
            self.norm_layer(self.last_input_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.last_input_channel,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=config.EXTRA.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if config.EXTRA.FINAL_CONV_KERNEL == 3 else 0)
        )
    def forward(self, x):
        feats = self.hrnet_forward(x)
        out = self.last_layer(feats)
        return out

def get_seg_model(cfg, **kwargs):
    model = HRNet(cfg,**kwargs)
    return model