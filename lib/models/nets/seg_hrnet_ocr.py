import torch
import torch.nn as nn
import torch.functional as F
from lib.models.backbones.basenet import BaseNet
from lib.models.modules.ocr_block import SpatialGather_Module,SpatialOCR_Module,OCRHead

import logging

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1

class HRNet_OCR(BaseNet):
    def __init__(self,config,last_input_channel = 720,**kwargs):
        super(HRNet_OCR,self).__init__(config,**kwargs)
        self.last_input_channel = last_input_channel
        ocr_mid_channels = config.MODEL.OCR.MID_CHANNELS
        ocr_key_channels = config.MODEL.OCR.KEY_CHANNELS
        self.ocr = OCRHead(config.DATASET.NUM_CLASSES, ocr_mid_channels, ocr_key_channels,
                           base_outchannel=self.last_input_channel)

        self.aux_head = nn.Sequential(
            nn.Conv2d(self.last_input_channel, self.last_input_channel,
                      kernel_size=1, stride=1, padding=0),
            self.norm_layer(self.last_input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.last_input_channel, config.DATASET.NUM_CLASSES,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
    def forward(self, x):
        feats = self.hrnet_forward(x)
        out_aux_seg = []

        # aux
        out_aux = self.aux_head(feats)
        # ocr
        out_ocr = self.ocr(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out_ocr)
        return out_aux_seg

def get_seg_model(cfg, **kwargs):
    model = HRNet_OCR(cfg,**kwargs)
    return model