import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


def steps(kernel_size):
    kernel_size -= 1
    out = torch.Tensor([[i, j] for j in range(-kernel_size // 2, kernel_size // 2 + 2)
                          for i in range(-kernel_size // 2, kernel_size // 2 + 2)])
    return torch.transpose(out, 0, 1)


class DeformConv2D(pl.LightningModule):
    def __init__(self, C, C_group,
                 batch,
                 kernel_size=3, stride=1,
                 padding=1, dilation=1):
        super(DeformConv2D, self).__init__()
        # Sizes
        self.G, self.C, self.C_group = C // C_group, C, C_group
        self.batch = batch

        # Layers
        self.conv = nn.Conv2d(C, 3 * C, kernel_size, stride, padding, dilation)
        self.softmax = nn.Softmax(dim=-1)

        # Params
        self.p0 = steps(kernel_size)
        self.w = nn.Parameter(torch.rand([self.G, C, C_group]), requires_grad=True)

    def x_offset(self, x):
        # Get offsets and modulation scalars
        out = self.conv(x)
        dp, dm = torch.split(out, [self.C * 2, self.C], dim=1)
        dm = self.softmax(dm) # apply softmax instead sigmoid

        # Apply bilinear interpolation to (x, x + p0 + dp)
        out_size = x.size()
        x = x.tile((1, 2, 1, 1))
        x_offset = x + self.p0 + dp
        out = F.interpolate(x_offset, out_size, mode='bilinear', align_corners=True)
        return out, dm

    def forward(self, x: torch.Tensor):
        x_offset, dm = self.x_offset(x)

        

        

