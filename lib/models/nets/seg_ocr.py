import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.basenet import BaseNet
from ..modules.ocr_block import OCRHead

import logging

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1

class OCRNet(BaseNet):
    def __init__(self, config, aux=False, **kwargs):
        super(OCRNet, self).__init__(config, **kwargs)
        ocr_mid_channels = config.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = config.MODEL.OCR.KEY_CHANNELS
        self.aux = aux
        self.ocr = OCRHead(config.DATASET.NUM_CLASSES, ocr_mid_channels, ocr_key_channels,norm_layer=self.norm_layer,
                           base_outchannel=self.base_outchannel)
        self.aux_layer = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1, bias=False),
                                       self.norm_layer(512),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(512, config.DATASET.NUM_CLASSES, 1,bias=True))

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        out = []

        if self.aux:
            aux_out = self.aux_layer(c3)
            out.append(aux_out)

        ocr_out = self.ocr(c4, aux_out)
        out.append(ocr_out)

        for i in range(len(out)):
            out[i] = F.interpolate(out[i], imsize, **self._up_kwargs)
        return tuple(out)


def get_seg_model(cfg, **kwargs):
    model = OCRNet(cfg, aux=True, **kwargs)
    return model


