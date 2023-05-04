import pytorch_lightning
import torch
from timm.models.layers import DropPath
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    def forward(self, input):
        """
        Applies layer normalization and return input with the same size

        :param input: input tensor of size (N, C, H, W)
        :type input: torch.Tensor
        """
        x = input.permute(0, 2, 3, 1)
        return super().forward(x).permute(0, 3, 1, 2)


def build_norm_layer(dim, norm_type='BN'):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim, eps=1e-5)
    elif norm_type == 'GN':
        return nn.GroupNorm(dim, dim, eps=1e-5)
    elif norm_type == 'LN':
        return LayerNorm2d(dim, eps=1e-6)
    elif norm_type == 'SBN':
        return nn.SyncBatchNorm(dim, eps=1e-5)
    else:
        raise ValueError('No such norm layer defined!')


def build_act_layer(act_type):
    if act_type == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_type == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_type == 'GELU':
        return nn.GELU()
    else:
        raise ValueError('No such activation defined!')


def split_with_ratio(x: torch.Tensor,
                     ratio: list[int],
                     dim: int = 0) -> list[torch.Tensor]:
    parts = sum(ratio)
    assert x.shape[dim] % parts == 0, f'Dimension={dim} {x.shape[dim]} ' \
                                      f'must be divisible in {ratio} ratio'
    ch_per_part = x.shape[dim] // parts
    return torch.split(x, [t * ch_per_part for t in ratio], dim)


class Stem(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 double=False,
                 norm_type='LN',
                 act_type='GELU'):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2 if double else out_channels, 3, 2, 1),
            build_norm_layer(out_channels // 2 if double else out_channels, norm_type),
            build_act_layer(act_type),
            # Output of first conv is half of second convolution (out_ch // 2 and out_ch)
            nn.Sequential(
                nn.Conv2d(out_channels // 2, out_channels, 3, 2, 1),
                build_norm_layer(out_channels, norm_type)) if double else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)


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


class FeatureDecompositionLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 act_type='GELU'):
        super().__init__()
        self.conn_1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.sigma = ElementScale(in_channels, init_value=1e-5, requires_grad=True)
        self.activation = build_act_layer(act_type)

    def forward(self, x):
        y = self.conn_1x1(x)
        return self.activation(y + self.sigma(y - self.GAP(y)))


class BranchLayer(nn.Module):
    def __init__(self,
                 ratio: list[int],
                 branches: list[nn.Module]):
        super().__init__()
        assert len(ratio) == len(branches)
        self.ratio = ratio
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        branch_inputs = split_with_ratio(x, self.ratio, 1)
        branch_outputs = []
        for i in range(len(self.branches)):
            branch_outputs.append(
                self.branches[i](branch_inputs[i])
            )
        return torch.cat(branch_outputs, dim=1)


class MultiOrderGatedAggregation(nn.Module):
    def __init__(self,
                 in_channels,
                 moga_ratio=None,
                 dilations=None,
                 act_type='SiLU'):
        super().__init__()
        self.aggregation_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            build_act_layer(act_type=act_type)
        )
        sum_ratio = sum(moga_ratio)
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=5,
                dilation=dilations[0],
                padding=(1 + 4 * dilations[0]) // 2,
                groups=in_channels
            ),
            BranchLayer(
                moga_ratio,
                [
                    nn.Identity(),
                    nn.Conv2d(in_channels * moga_ratio[1] // sum_ratio,
                              in_channels * moga_ratio[1] // sum_ratio,
                              kernel_size=5,
                              padding=(1 + 4 * dilations[1]) // 2,
                              dilation=dilations[1],
                              groups=in_channels * moga_ratio[1] // sum_ratio),
                    nn.Conv2d(in_channels * moga_ratio[2] // sum_ratio,
                              in_channels * moga_ratio[2] // sum_ratio,
                              kernel_size=7,
                              padding=(1 + 6 * dilations[2]) // 2,
                              dilation=dilations[2],
                              groups=in_channels * moga_ratio[2] // sum_ratio)
                ]
            ),
            nn.Conv2d(in_channels, in_channels, 1),
            build_act_layer(act_type)
        )
        self.pw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1,
        )

    def forward(self, x):
        return self.pw_conv(self.aggregation_branch(x) * self.context_branch(x))


class SpatialAggregationLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 fd_act_type='GELU',
                 moga_act_type='SiLU',
                 moga_ratio=None,
                 dilations=None):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.feature_decompose = FeatureDecompositionLayer(in_channels, fd_act_type)
        self.moga = MultiOrderGatedAggregation(in_channels, moga_ratio,
                                               dilations, moga_act_type)

    def forward(self, x):
        return x + self.moga(self.feature_decompose(self.norm(x)))


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


class ChannelAggregationLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 scale,
                 drop_rate,
                 act_type='GELU'):
        super().__init__()
        hidden_dim = int(in_channels * scale)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, hidden_dim,
                      kernel_size=3, padding=1, groups=hidden_dim),
            build_act_layer(act_type),
            nn.Dropout(drop_rate),
            ChannelAggregation(hidden_dim),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        return x + self.layer(x)


class MogaBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 moga_ratio,
                 dilations,
                 ffn_scale,
                 drop_path_rate,
                 drop_rate,
                 ffn_act_type,
                 fd_act_type,
                 moga_act_type):
        super().__init__()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.attn_block = SpatialAggregationLayer(in_channels,
                                                  fd_act_type,
                                                  moga_act_type,
                                                  moga_ratio,
                                                  dilations)
        self.ffn_block = ChannelAggregationLayer(in_channels,
                                                 ffn_scale,
                                                 drop_rate,
                                                 ffn_act_type, )

    def forward(self, x):
        residual = x
        x = self.attn_block(x)
        x = residual + self.drop_path(x)
        residual = x
        x = self.ffn_block(x)
        return residual + self.drop_path(x)


class MogaNet(pytorch_lightning.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_indices: list[int],
                 depths: list[int],
                 widths: list[int],
                 stem_act_type: str,
                 stem_norm_type: str,
                 moga_ratio: list[int],
                 moga_dilations: list[int],
                 drop_path_rate: float,
                 drop_rate: float,
                 ffn_scales: list[float],
                 ffn_act_type: str,
                 fd_act_type: str,
                 moga_act_type: str):
        super().__init__()
        self.use_middle_steps = len(out_indices) != 0
        self.out_indices = out_indices
        self.stages = nn.ModuleList([
            nn.Sequential(
                Stem(in_channels=in_channels if i == 0 else widths[i - 1],
                     out_channels=widths[i],
                     double=i == 0,
                     act_type=stem_act_type,
                     norm_type=stem_norm_type),
                *[MogaBlock(in_channels=widths[i],
                            moga_ratio=moga_ratio,
                            dilations=moga_dilations,
                            ffn_scale=ffn_scales[i],
                            drop_path_rate=drop_path_rate,
                            drop_rate=drop_rate,
                            ffn_act_type=ffn_act_type,
                            fd_act_type=fd_act_type,
                            moga_act_type=moga_act_type) for _ in range(depths[i])]
            )
            for i in range(len(depths))
        ])

    def forward(self, x):
        seq_out = []
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx in self.out_indices:
                seq_out.append(x.contiguous())
        if self.use_middle_steps:
            return seq_out
        return x
