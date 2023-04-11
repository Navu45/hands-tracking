import torch
from timm.models.layers import DropPath
from torch import nn

from feature_extractor.deform_conv import DCNv3


class Stem(nn.Module):
    def __init__(self,
                 in_ch=3,
                 out_ch=64):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, 2, 1),
            nn.LayerNorm(out_ch // 2),
            nn.GELU(),
            # Output of first conv is half of second convolution (out_ch // 2 and out_ch)
            nn.Conv2d(out_ch // 2, out_ch, 3, 2, 1),
            nn.LayerNorm(out_ch),
        )

    def forward(self, x):
        return self.layer(x)


class Downsample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.layer = nn.Sequential(
            # resize channels and downsample input map by 2
            nn.Conv2d(in_ch, in_ch * 2, 3, 2, 1),
            nn.LayerNorm(in_ch * 2)
        )

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    def __init__(self,
                 in_ch: int,
                 hidden_features=None,
                 drop_rate=0.):
        super().__init__()
        hidden_features = hidden_features or in_ch
        self.fc1 = nn.Linear(in_ch, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_ch)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

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
                 group: int,
                 drop_path: float,
                 drop_rate: float,
                 mlp_ratio: float,
                 layer_scale: float,
                 post_norm=True):
        super().__init__()
        self.post_norm = post_norm
        print(in_ch)
        self.dcn = DCNv3(channels=in_ch,
                         group=group)
        self.norm1 = nn.LayerNorm(in_ch)
        self.mlp = MLP(in_ch=in_ch,
                       hidden_features=int(in_ch * mlp_ratio),
                       drop_rate=drop_rate)
        self.norm2 = nn.LayerNorm([int(in_ch * mlp_ratio)])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(in_ch),
                                   requires_grad=True)
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(in_ch),
                                   requires_grad=True)

    def forward(self, x):
        if not self.post_norm:
            x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
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
                 post_norm: bool,
                 use_downsample: bool
                 ):
        super().__init__()
        self.post_norm = post_norm
        self.use_downsample = use_downsample

        self.blocks = nn.ModuleList([
            BasicBlock(in_ch=in_ch,
                       group=group,
                       drop_path=drop_path,
                       drop_rate=drop_rate,
                       mlp_ratio=mlp_ratio,
                       layer_scale=layer_scale,
                       post_norm=post_norm)
            for _ in range(depth)
        ])
        if use_downsample:
            self.downsample = Downsample(in_ch)


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.post_norm:
            x = self.norm(x)
        if self.use_downsample:
            return self.downsample(x), x
        return None, x


class InternImage(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embedding_ch=64,
                 out_indices=None,
                 depths=None,
                 groups=None,
                 post_norm=True,
                 drop_path=0.,
                 drop_rate=0.,
                 mlp_ratio=0.,
                 layer_scale=0.4):
        super().__init__()
        self.out_indices = out_indices
        self.stem = Stem(in_channels, embedding_ch)
        self.pos_drop = nn.Dropout(drop_rate)
        self.levels = nn.ModuleList([
            BasicLayer(
                embedding_ch * 2 ** i,
                depths[i],
                groups[i],
                drop_path,
                drop_rate,
                mlp_ratio,
                layer_scale,
                post_norm,
                i != len(depths) - 1
            )
            for i in range(len(depths))
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


if __name__ == '__main__':
    model = InternImage(
        in_channels=3,
        embedding_ch=64,
        out_indices=[0, 1, 2, 3],
        depths=[3, 3, 3, 3],
        groups=[1, 2, 4, 8],
        post_norm=True,
        drop_path=0.3,
        drop_rate=0.25,
        mlp_ratio=1.0,
        layer_scale=.8
    )

    test_tensor = torch.rand(1, 3, 224, 224)
    output = model(test_tensor)
    print(output)
