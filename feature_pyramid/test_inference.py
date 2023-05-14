import torch
from pytorch_lightning.cli import LightningCLI

from feature_pyramid.model import TransformerFCN


class TransformerFCNModule(TransformerFCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def cli_main():
    # Init model with YAML config via command line
    cli = LightningCLI(TransformerFCNModule, run=False)
    tensor = [
        torch.rand(16, 64, 32, 32),
        torch.rand(16, 96, 16, 16),
        torch.rand(16, 192, 8, 8),
    ]
    print(cli.model)
    # Test model on random data
    print(cli.model(tensor))
    # Save model to torchscript file
    cli.model.to_torchscript('data/models/torch-script/transformer_fcn.pt')


if __name__ == '__main__':
    cli_main()
