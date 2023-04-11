import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from pytorch_lightning.cli import LightningCLI
from torchvision.transforms import transforms

from data import CustomDataModule
from torchvision.datasets import MNIST

from feature_extractor import InternImage
from feature_extractor.deform_conv.modules import build_norm_layer


class MNIST_dataset_configure:
    def __call__(self, root, train=False, val=False, download=False,
                 test=False, predict=False, transform=None):
        MNIST(root, download=True)
        return MNIST(root,
                     download=download,
                     train=(train or val and not test) or not predict,
                     transform=transform)


class TestClassifier(pl.LightningModule):
    def __init__(self,
                 cls_scale,
                 imsize,
                 num_classes,
                 backbone_params: dict):
        super().__init__()
        num_stages = len(backbone_params['depths'])
        num_features = int(backbone_params['embedding_ch'] * 2 ** (num_stages - 1))
        imsize //= 2 ** (num_stages - 2)
        print(imsize)
        self.backbone = InternImage(**backbone_params)
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features,
                      int(num_features * cls_scale),
                      kernel_size=1,
                      bias=False),
            build_norm_layer(int(num_features * cls_scale), 'BN',
                             'channel_first', 'channels_first'),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(int(num_features * cls_scale) * imsize, num_features),
            nn.GELU(),
            nn.Linear(num_features, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metric = torchmetrics.F1Score(task='multiclass', threshold=0.7, num_classes=10, average='weighted')

    def forward(self, x):
        features = self.backbone(x, False)
        print(features.size())
        pred = self.classifier(features)
        print(pred.size())
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


def cli_main():
    cli = LightningCLI(TestClassifier, CustomDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    cli_main()
