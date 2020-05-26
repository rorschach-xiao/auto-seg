import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize

import logging

from ..tools.bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace,GroupNorm2d
from .hrnet import hrnet64,hrnet48,hrnet32,hrnet18
from .resnest import resnest50,resnest101,resnest200
from .resnet import resnet50,resnet101,resnet152

import torch.nn.functional as F


logger = logging.getLogger(__name__)

def get_backbone(name):
    backbone = {
        "hrnet18": hrnet18,
        "hrnet32": hrnet32,
        "hrnet48": hrnet48,
        "hrnet64": hrnet64,

        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,

        "resnest50": resnest50,
        "resnest101": resnest101,
        "resnest200": resnest200,

    }
    if name not in backbone:
        raise NotImplementedError("backbone {} not implement yet!".format(name))
    else:
        return backbone[name]


class BaseNet(nn.Module):
    def __init__(self,config,**kwargs):
        super(BaseNet, self).__init__()
        self.nclass = config.DATASET.NUM_CLASSES
        self.backbone = config.MODEL.BACKBONE
        self.dilated = config.MODEL.DILATION
        self.multi_grid = config.MODEL.MULTI_GRID
        self.multi_dilation = [4, 8, 16]
        if config.TRAIN.BN_TYPE =="BN":
            self.norm_layer = BatchNorm2d
        elif config.TRAIN.BN_TYPE == "GN":
            self.norm_layer = GroupNorm2d
        else:
            raise ValueError("only support sync-bn and gn")

        self.model_path = config.MODEL.PRETRAINED
        self.ispretrained = (config.MODEL.PRETRAINED != "")
        self.base_outchannel = 2048
        self.deep_base = config.MODEL.DEEPBASE
        self.spm_on = config.MODEL.SPM

        self._up_kwargs = {'mode': 'bilinear', 'align_corners': config.MODEL.ALIGN_CORNERS}

        self.pretrained = get_backbone(self.backbone)(pretrained=self.ispretrained, dilated=self.dilated,
                                              norm_layer=self.norm_layer, model_path=self.model_path,
                                              multi_grid=self.multi_grid, multi_dilation=self.multi_dilation,deep_base=self.deep_base,spm_on=self.spm_on)

    def base_forward(self,x):
        '''
        use for resnet and resnest forward
        '''
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def hrnet_forward(self,x):
        '''
        use for hrnet forward
        '''
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.conv2(x)
        x = self.pretrained.bn2(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.layer1(x)

        x_list = []
        for i in range(self.pretrained.stage2_cfg['NUM_BRANCHES']):
            if self.pretrained.transition1[i] is not None:
                x_list.append(self.pretrained.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.pretrained.stage2(x_list)

        x_list = []
        for i in range(self.pretrained.stage3_cfg['NUM_BRANCHES']):
            if self.pretrained.transition2[i] is not None:
                if i < self.pretrained.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.pretrained.transition2[i](y_list[i]))
                else:
                    x_list.append(self.pretrained.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.pretrained.stage3(x_list)

        x_list = []
        for i in range(self.pretrained.stage4_cfg['NUM_BRANCHES']):
            if self.pretrained.transition3[i] is not None:
                if i < self.pretrained.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.pretrained.transition3[i](y_list[i]))
                else:
                    x_list.append(self.pretrained.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.pretrained.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           **self._up_kwargs)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           **self._up_kwargs)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           **self._up_kwargs)

        feats = torch.cat([x[0], x1, x2, x3], 1)

        return feats


