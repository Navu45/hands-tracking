import os

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.cli import LightningCLI
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from data import CustomDataModule
from model import MogaNet


class MNIST_dataset_configure:
    def __call__(self, root, train=False, val=False,
                 test=False, predict=False, transform=None):
        MNIST(root, download=True)
        dataset = MNIST(root,
                        train=(train or val and not test) or not predict,
                        transform=transform)
        dataset.data = dataset.data[:3000]
        return dataset


class TestClassifier(pl.LightningModule):
    def __init__(self,
                 imsize,
                 num_classes,
                 backbone_params: dict):
        super().__init__()
        num_stages = len(backbone_params['depths'])
        num_features = int(backbone_params['widths'][-1])
        imsize //= 2 ** (num_stages - 1)
        self.backbone = MogaNet(**backbone_params)
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features,
                      num_features // 4,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features // 4),
            nn.GELU(),
            nn.Conv2d(num_features // 4,
                      num_features // 16,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features // 16),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(num_features // 16 * imsize, num_features // 16),
            nn.GELU(),
            nn.Linear(num_features // 16, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metric = torchmetrics.F1Score(task='multiclass', threshold=0.7, num_classes=10, average='weighted')
        self.to(memory_format=torch.channels_last)
        self.save_hyperparameters()

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        features = self.backbone(x)
        pred = self.classifier(features[-1])
        return pred

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.forward(x)
        x_hat.unsqueeze(-1)
        loss = self.criterion(x_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.forward(x)
        x_hat.unsqueeze(-1)
        loss = self.metric(x_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss


class MNISTransform(nn.Module):
    def __init__(self, size=256):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.7,), (0.7,))
        ])

    def forward(self, x):
        return self.transform(x)


def cli_main(ckpt_path="data/models/backbone_mnist.ckpt"):
    cli = LightningCLI(TestClassifier, CustomDataModule, run=False)
    if os.path.exists(ckpt_path):
        cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    else:
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)



if __name__ == "__main__":
    cli_main()
