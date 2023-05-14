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

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        img_features = self.backbone(x)
        outputs = self.head(img_features)
        loss = self.criterion(outputs, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        img_features = self.backbone(x)
        outputs = self.head(img_features)
        print(torch.Tensor(outputs))
        loss = self.criterion(outputs, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        img_features = self.backbone(x)
        outputs = self.head(img_features)
        print(torch.Tensor(outputs))
        loss = self.criterion(outputs, y)
        self.log("test_loss", loss)


