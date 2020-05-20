import torch
import torch.nn as nn
import torch.functional as F


__all__ = ['PyramidPooling']

class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)
        self.inchannel = in_channels
        self.out_channels = int(in_channels/4)
        self.conv1_ = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                norm_layer(self.out_channels),
                                nn.ReLU(True))
        self.conv2_ = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                norm_layer(self.out_channels),
                                nn.ReLU(True))
        self.conv3_ = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                norm_layer(self.out_channels),
                                nn.ReLU(True))
        self.conv4_ = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                norm_layer(self.out_channels),
                                nn.ReLU(True))
        self.out_layer = nn.Sequential(
            nn.Conv2d(self.out_channels*4+self.inchannel, self.out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(self.out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1_(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2_(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3_(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4_(self.pool4(x)), (h, w), **self._up_kwargs)
        out = torch.cat((x, feat1, feat2, feat3, feat4), 1)
        out = self.out_layer(out) # batchsize * 512 * h * w
        return out