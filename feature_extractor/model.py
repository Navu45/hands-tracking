import torch
from timm.models.layers import DropPath
from torch import nn

from feature_extractor.deform_conv.modules import DCNv3_pytorch, build_norm_layer, to_channels_first


class Stem(nn.Module):
    def __init__(self,
                 in_ch=3,
                 out_ch=64):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, 2, 1),
            build_norm_layer(out_ch // 2, 'LN', 'channels_first', 'channels_first'),
            nn.GELU(),
            # Output of first conv is half of second convolution (out_ch // 2 and out_ch)
            nn.Conv2d(out_ch // 2, out_ch, 3, 2, 1),
            build_norm_layer(out_ch, 'LN', 'channels_first', 'channels_last')
        )

    def forward(self, x):
        return self.layer(x)


class Downsample(nn.Module):
    def __init__(self,
                 in_ch: int, ):
        super().__init__()
        self.layer = nn.Sequential(
            # resize channels and downsample input map by 2
            nn.Conv2d(in_ch, in_ch * 2, 3, 2, 1),
            build_norm_layer(in_ch * 2, 'LN', 'channels_first', 'channels_last'),
        )

    def forward(self, x):
        return self.layer(x.permute(0, 3, 1, 2))


class MLP(nn.Module):
    def __init__(self,
                 in_ch: int,
                 hidden_features=None,
                 drop_rate=0.):
        super().__init__()
        hidden_features = hidden_features or in_ch
        self.layer = nn.Sequential(
            nn.Linear(in_ch, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_ch),
            nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 group: int,
                 drop_path: float,
                 drop_rate: float,
                 mlp_ratio: float,
                 layer_scale: float):
        super().__init__()
        self.dcn = DCNv3_pytorch(channels=in_ch,
                                 group=group,
                                 norm_layer="BN")
        self.norm1 = nn.LayerNorm(in_ch)
        self.mlp = MLP(in_ch=in_ch,
                       hidden_features=int(in_ch * mlp_ratio),
                       drop_rate=drop_rate)
        self.norm2 = nn.LayerNorm(in_ch)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(in_ch),
                                   requires_grad=True)
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(in_ch),
                                   requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
        x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 in_ch: int,
                 depth: int,
                 group: int,
                 drop_path: float,
                 drop_rate: float,
                 mlp_ratio: float,
                 layer_scale: float,
                 use_downsample: bool
                 ):
        super().__init__()
        self.use_downsample = use_downsample

        self.blocks = nn.ModuleList([
            BasicBlock(in_ch=in_ch,
                       group=group,
                       drop_path=drop_path,
                       drop_rate=drop_rate,
                       mlp_ratio=mlp_ratio,
                       layer_scale=layer_scale)
            for _ in range(depth)
        ])
        if use_downsample:
            self.downsample = Downsample(in_ch)
        else:
            self.channels_first = to_channels_first()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.use_downsample:
            return self.downsample(x), x
        return None, self.channels_first(x)


class InternImage(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embedding_ch=64,
                 out_indices=None,
                 depths=None,
                 groups=None,
                 drop_path=0.,
                 drop_rate=0.,
                 mlp_ratio=0.,
                 layer_scale=0.4):
        super().__init__()
        self.out_indices = out_indices
        self.stem = Stem(in_channels,
                         embedding_ch)
        self.dropout = nn.Dropout(drop_rate)
        self.levels = nn.ModuleList([
            BasicLayer(
                embedding_ch * 2 ** i,
                depths[i],
                groups[i],
                drop_path,
                drop_rate,
                mlp_ratio,
                layer_scale,
                i != len(depths) - 1
            )
            for i in range(len(depths))
        ])
        print(self)

    def forward(self, x, use_middle_steps=True):
        x = self.stem(x)
        x = self.dropout(x)

        seq_out = []
        _x = None
        for level_idx, level in enumerate(self.levels):
            x, _x = level(x)
            if use_middle_steps and level_idx in self.out_indices:
                seq_out.append(_x.contiguous())
        if use_middle_steps:
            return seq_out.reverse()
        return _x
