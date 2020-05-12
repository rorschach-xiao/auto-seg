import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from lib.models.modules.psp_block import PyramidPooling
from lib.models.backbones.basenet import BaseNet

logger = logging.getLogger(__name__)

class PSPNet(BaseNet):
    def __init__(self,config,aux=True,**kwargs):
        super(PSPNet,self).__init__(config,**kwargs)
        self.inchannel = self.base_outchannel
        self.pyramid_pooling = PyramidPooling(self.inchannel,self.norm_layer,self._up_kwargs)
        self.cls_layer = nn.Conv2d(int(self.inchannel/4),self.nclass,kernel_size=1,padding=0,bias=False)
        self.aux_layer = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                self.norm_layer(256),
                                nn.ReLU(),
                                nn.Dropout2d(0.1, False),
                                nn.Conv2d(256, self.nclass, 1))
        for n, m in self.pretrained.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.pretrained.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        self.aux = aux

    def forward(self, x):
            imsize = x.size()[2:]
            if 'hrnet' in self.backbone:
                c4 = self.hrnet_forward(x)
                c3 = c4
            else:
                _, _, c3, c4 = self.base_forward(x)
            out = []
            out.append(self.cls_layer(self.pyramid_pooling(c4)))
            if self.aux:  # auxiliary branch
                out.append(self.aux_layer(c3))
            for i in range(len(out)):
                out[i] = F.interpolate(out[i], imsize, **self._up_kwargs)
            return tuple(out)

    def get_seg_model(cfg, **kwargs):
        model = PSPNet(cfg, aux=True, **kwargs)
        return model