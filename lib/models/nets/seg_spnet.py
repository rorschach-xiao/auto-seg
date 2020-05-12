import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from lib.models.modules.sp_block import StripPooling
from lib.models.backbones.basenet import BaseNet

logger = logging.getLogger(__name__)

class SPNet(BaseNet):
    def __init__(self, config, aux=True, **kwargs):
        super(SPNet, self).__init__(config, **kwargs)
        self.mpm = SPHead(2048, self.nclass, self.norm_layer, self._up_kwargs)
        self.aux = aux
        if self.aux:
            self.aux_layer = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                           self.norm_layer(256),
                                           nn.ReLU(),
                                           nn.Dropout2d(0.1, False),
                                           nn.Conv2d(256, self.nclass, 1))
        if self.deep_base:
            self.pretrained.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                self.norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                self.norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.pretrained.bn1 = self.norm_layer(128)
            self.pretrained.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, h, w = x.size()
        if self.backbone=='hrnet':
            c4 = self.hrnet_forward(x)
            c3 = c4
        else:
            _,_,c3,c4 = self.base_forward(x)
        out = []
        x = self.mpm(c4)
        x = F.interpolate(x, (h, w), **self._up_kwargs)
        out.append(x)
        if self.aux:
            auxout = self.aux_layer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            out.append(auxout)

        return tuple(out)

class SPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True)
                                         )
        self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                                         norm_layer(inter_channels // 2),
                                         nn.ReLU(True),
                                         nn.Dropout2d(0.1, False),
                                         nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x

def get_seg_model(cfg, **kwargs):
    model = SPNet(cfg, aux=True, **kwargs)
    return model