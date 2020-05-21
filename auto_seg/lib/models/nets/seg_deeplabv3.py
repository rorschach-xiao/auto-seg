from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from ..modules.aspp_block import ASPP
from ..backbones.basenet import BaseNet


up_kwargs = {'mode': 'bilinear', 'align_corners': False}

logger = logging.getLogger(__name__)


class DeepLabV3(BaseNet):
    def __init__(self,config,aux = True,**kwargs):
        super(DeepLabV3, self).__init__(config,**kwargs)
        self.atrous_rate = config.MODEL.ATROUS_RATE #  [6,12,18] for stride 16, [12,24,36] for stride 8
        self.aspp = ASPP(num_classes=self.nclass,atrous_rate=self.atrous_rate,inchannel=2048,norm_layer=self.norm_layer)
        self.aux = aux
        self.aux_layer = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                       self.norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(256, self.nclass, 1))
    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        imsize = x.size()[2:]
        if 'hrnet' in self.backbone:
            c4 = self.hrnet_forward(x)
            c3 = c4
        else:
            _, _, c3, c4 = self.base_forward(x)

        out = []
        if self.aux:
            aux_out = self.aux_layer(c3)
            aux_out = F.interpolate(aux_out,size=imsize,**self._up_kwargs)
            out.append(aux_out)

        output = self.aspp(c4) # (shape: (batch_size, num_classes, h/16, w/16))
        output = F.interpolate(output, size=imsize,**self._up_kwargs) # (shape: (batch_size, num_classes, h, w))
        out.append(output)

        return tuple(out)

def get_seg_model(cfg, **kwargs):
    model = DeepLabV3(cfg, aux=True,**kwargs)
    return model