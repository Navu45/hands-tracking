import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.cli import LightningCLI
from torchvision.transforms import transforms

from data import CustomDataModule
from torchvision.datasets import MNIST
from feature_extractor import InternImage, MLP


def configure_mnist_dataset(root, train=False, val=False, download=False,
                            test=False, predict=False, transform=None):
    return MNIST(root,
                 download=download,
                 train=(train or val and not test) or not predict,
                 transform=transform)


class TestClassifier(pl.LightningModule):
    def __init__(self, **backbone_params):
        super().__init__()
        self.backbone = InternImage(**backbone_params)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        loss = F.mse_loss(x_hat, x)
        return loss




class MnistTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.7,), (0.7,))
        ])

    def forward(self, x):
        return self.transform(x)



def cli_main():
    cli = LightningCLI(TestClassifier, CustomDataModule)


if __name__ == "__main__":
    cli_main()
