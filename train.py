import pytorch_lightning as pl
import torch
import torchmetrics
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class HandKeypointDetector(pl.LightningModule):
    def __init__(self, backbone, head, config):
        super().__init__()
        self.backbone = torch.jit.load('/kaggle/input/models-diploma')
        self.head = torch.jit.load('/kaggle/input/models-diploma/transformer_fcn.pt')
        self.config = config
        self.save_hyperparameters()
        self.criterion = torchmetrics.MeanSquaredError()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.config['optimizer'])
        scheduler = CosineAnnealingLR(optimizer, **self.config['scheduler'])
        return optimizer, scheduler

    def training_step(self):
        pass



