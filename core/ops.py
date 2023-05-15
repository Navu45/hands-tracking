import copy

import torch
from torch import nn


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def build_norm_layer(dim, norm_type='BN'):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim, eps=1e-5)
    elif norm_type == 'GN':
        return nn.GroupNorm(dim, dim, eps=1e-5)
    elif norm_type == 'LN':
        return LayerNorm2d(dim, eps=1e-5, data_format='channels_first')
    else:
        return nn.Identity()


def build_act_layer(act_type):
    if act_type == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_type == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_type == 'GELU':
        return nn.GELU()
    else:
        return nn.Identity()


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
                 dense_block):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1)
        self.dense_block = dense_block

    def forward(self, x):
        return self.branch1(x) + self.dense_block(self.branch2(x))


class ElementScale(nn.Module):
    def __init__(self,
                 channels,
                 init_value=0.,
                 requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(1, channels, 1, 1),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregation(nn.Module):
    def __init__(self,
                 in_channels,
                 act_type='GELU'):
        super().__init__()
        self.CA = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            build_act_layer(act_type)
        )
        self.sigma = ElementScale(in_channels, requires_grad=True)

    def forward(self, x):
        return x + self.sigma(x - self.CA(x))


class ConvolutionalMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 mlp_ratio,
                 drop_rate,
                 act_type='GELU'):
        super().__init__()
        hidden_dim = int(in_channels * mlp_ratio)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            SeparableConv2d(in_channels, hidden_dim,
                            kernel_size=3, padding=1,
                            depthwise_first=False),
            build_act_layer(act_type),
            nn.Dropout(drop_rate),
            ChannelAggregation(hidden_dim),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        return x + self.layer(x)
