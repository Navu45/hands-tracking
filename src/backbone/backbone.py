import lightning.pytorch as pl
from torch import nn


class BackBone(pl.LightningModule):
    def __init__(self, size, *hyperparams):
        super().__init__()
        self.size = size
        self.H, self.W = size
        self.Ci, self.C_, self.L1, self.L3 = hyperparams
        self.Li = 1
        

    def CHW(self):
        return [self.Ci, self.H, self.W]

    def stem(self):
        stem_layer = nn.Sequential(
            nn.Conv2d(3, self.Ci, 3, 2, 1),
            nn.LayerNorm([self.Ci, self.H // 2, self.W // 2]),
            nn.GELU(),
            nn.Conv2d(self.Ci, self.Ci, 3, 2, 1),
            nn.LayerNorm([self.Ci, self.H // 4, self.W // 4]),
        )
        self.H, self.W = self.H // 4, self.W // 4
        self.Ci *= 2
        return stem_layer

    def downsampling(self):
        self.H, self.W = self.H // 2, self.W // 2
        downsampler = nn.Sequential(
            nn.Conv2d(self.Ci, self.Ci * 2, 3, 2, 1),
            nn.LayerNorm(self.CHW()),
        )
        self.Ci *= 2
        return downsampler

    def basic_block(self):
        return nn.Sequential(
            deform_conv(),
            nn.LayerNorm(self.CHW()),
        ), nn.Sequential(
            nn.Conv2d(self.Ci, self.Ci, 3, 2, 1),
            nn.LayerNorm(self.CHW())
        )

    def stage(self):
        stage_i = nn.ModuleDict({
            "block_%d" % i: nn.ModuleDict({

            }) for i in range(1, 5)
        })
