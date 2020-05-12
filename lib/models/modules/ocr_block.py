import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.tools.bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
from torch.autograd import Variable

logger = logging.getLogger(__name__)
ALIGN_CORNERS = True

__all__ = ['SpatialGather_Module','SpatialOCR_Module','SpatialOCR_ASP_Module','OCRHead']

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context

class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class SpatialOCR_Context(nn.Module):
    """
    Implementation of the FastOC module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self, in_channels, key_channels, scale=1, dropout=0, bn_type=None, ):
        super(SpatialOCR_Context, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type=bn_type)

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        return context

class SpatialOCR_ASP_Module(nn.Module):
    def __init__(self, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(SpatialOCR_ASP_Module, self).__init__()

        self.context = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                     SpatialOCR_Context(in_channels=hidden_features,
                                                        key_channels=hidden_features//2, scale=1, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        self.object_head = SpatialGather_Module(num_classes)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        feat1 = self.context[2](feat1, proxy_feats)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output

class OCRHead(nn.Module):
    def __init__(self,nclass,ocr_mid_channels,ocr_key_channels,base_outchannel=2048):
        super(OCRHead,self).__init__()
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(base_outchannel, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=relu_inplace),
        )

        self.ocr_gather_head = SpatialGather_Module(nclass)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, nclass, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x,x_aux):

        feats = self.conv3x3_ocr(x)
        context = self.ocr_gather_head(feats, x_aux)
        feats = self.ocr_distri_head(feats, context)
        ocr_out = self.cls_head(feats)


        return ocr_out