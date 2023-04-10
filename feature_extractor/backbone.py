from torch import nn
from torch.nn.modules.utils import _pair
from deform_conv.modules import DCNv3, to_channels_last
from timm.models.layers import DropPath


class Stem(nn.Module):
    def __init__(self,
                 in_ch=3,
                 out_ch=64,
                 imsize=224):
        super().__init__()
        imsize = _pair(imsize)
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, 2, 1),
            nn.LayerNorm([out_ch // 2, *imsize]),
            nn.GELU(),
            # Output of first conv is half of second convolution (out_ch // 2 and out_ch)
            nn.Conv2d(out_ch // 2, out_ch, 3, 2, 1),
            nn.LayerNorm([out_ch, *imsize]),
            to_channels_last()
        )

    def forward(self, x):
        return self.layer(x)


class Downsample(nn.Module):
    def __init__(self,
                 in_ch: int,
                 imsize: tuple[int, int]):
        super().__init__()
        self.layer = nn.Sequential(
            # resize channels and downsample input map by 2
            nn.Conv2d(in_ch, in_ch * 2, 3, 2, 1),
            nn.LayerNorm([in_ch * 2, *imsize])
        )

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    def __init__(self,
                 in_ch: int,
                 hidden_features=None,
                 drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_ch
        self.fc1 = nn.Linear(in_ch, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_ch)
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 imsize: tuple[int, int],
                 group: int,
                 drop_path=0.,
                 drop=0.,
                 mlp_ratio=0.,
                 post_norm=True):
        super().__init__()
        self.post_norm = post_norm

        self.dcn = DCNv3(channels=in_ch,
                         group=group)
        self.norm1 = nn.LayerNorm([in_ch, *imsize])
        self.mlp = MLP(in_ch=in_ch,
                       hidden_features=int(in_ch * mlp_ratio),
                       drop=drop)
        self.norm2 = nn.LayerNorm([int(in_ch * mlp_ratio), *imsize])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if not self.post_norm:
            x = x + self.drop_path(self.dcn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.norm1(self.dcn(x)))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class Stage(nn.Module):
    def __init__(self,
                 in_ch: int,
                 imsize: int,
                 depth: int,
                 group: int,
                 dropout_params: dict,
                 post_norm: bool,
                 use_downsample: bool
                 ):
        super().__init__()
        self.post_norm = post_norm
        imsize = _pair(imsize)

        self.blocks = nn.ModuleList([
            BasicBlock(in_ch=in_ch,
                       imsize=imsize,
                       group=group,
                       **dropout_params,
                       post_norm=post_norm)
            for _ in range(depth)
        ])
        if use_downsample:
            self.blocks.append(
                Downsample(
                    in_ch,
                    imsize
                )
            )
        if self.post_norm:
            self.norm = nn.LayerNorm([in_ch, *imsize])


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class InternImage(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embedding_ch=64,
                 imsize=224,
                 out_indices=None,
                 depths=None,
                 groups=None,
                 dropout_params=None,
                 post_norm=True):
        super().__init__()

        self.stem = Stem(in_channels, embedding_ch, imsize)
        self.pos_drop = nn.Dropout(p=dropout_params['drop_rate'])
        self.levels = nn.ModuleList([
            Stage(
                embedding_ch * i,
                imsize // (2 ** (i + 1)),
                depths[i],
                groups[i],
                dropout_params,
                post_norm,
                i != len(depths)
            )
            for i in range(1, len(depths) + 1)
        ])

    def forward(self, x):
        x = self.stem(x)
        x = self.dropout(x)

        seq_out = []
        for level_idx, level in enumerate(self.levels):
            x, _x = level(x)
            if level_idx in self.out_indices:
                seq_out.append(_x.permute(0, 3, 1, 2).contiguous())
        return seq_out
