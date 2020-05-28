import torch
import functools
import torch.nn as nn

if torch.__version__.startswith('0'):
    from .sync_bn.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    GroupNorm2d = nn.GroupNorm
    relu_inplace = True


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, norm_layer=BatchNorm2d, **kwargs):
        return nn.Sequential(
            norm_layer(num_features),
            nn.ReLU()
        )
    @staticmethod
    def BN(num_features,norm_layer=BatchNorm2d,num_group=8,**kwargs):
        if norm_layer == BatchNorm2d or norm_layer == torch.nn.BatchNorm2d:
            return norm_layer(num_features)
        elif norm_layer == GroupNorm2d:
            return norm_layer(num_group,num_features)
        else:
            raise NotImplementedError("only support sync-bn,bn and gn")


