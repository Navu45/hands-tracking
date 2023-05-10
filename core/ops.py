import torch
from torch import nn


def build_norm_layer(dim, norm_type='BN'):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim, eps=1e-5)
    elif norm_type == 'GN':
        return nn.GroupNorm(dim, dim, eps=1e-5)
    else:
        raise ValueError('No such normalization layer defined!')


def build_act_layer(act_type):
    if act_type == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_type == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_type == 'GELU':
        return nn.GELU()
    else:
        raise ValueError('No such activation defined!')


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1,
                 padding=0, dilation=1,
                 norm_type=None,
                 depthwise_first=True):
        """
        Separable convolution divided in depthwise and pointwise parts.
        Normalization can be specified between two layers (`norm_type` param).
        Args:
            norm_type (str): normalization type between depthwise conv and pointwise conv (BN, LN, etc).
        """
        super().__init__()
        if depthwise_first:
            self.block = nn.Sequential(
                # Depthwise part
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                          padding=padding, stride=stride,
                          dilation=dilation, groups=in_channels,
                          bias=norm_type is None),
                build_norm_layer(in_channels, norm_type) if norm_type is not None else nn.Identity(),
                # Pointwise part
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.block = nn.Sequential(
                # Pointwise part
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                build_norm_layer(out_channels, norm_type) if norm_type is not None else nn.Identity(),
                # Depthwise part
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=padding, stride=stride,
                          dilation=dilation, groups=in_channels,
                          bias=norm_type is None),
            )

    def forward(self, x):
        return self.block(x)


class CSPBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 dense_block, norm_type, act_type,
                 kernel_size=1, stride=1, padding=0, dilation=1, ):
        super().__init__()
        self.branch1 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation),
            build_norm_layer(norm_type),
            build_act_layer(act_type)

        )
        self.branch2 = nn.Sequential(
            SeparableConv2d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation),
            build_norm_layer(norm_type),
            build_act_layer(act_type),
            dense_block
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)  # element-wise addition instead of concat
