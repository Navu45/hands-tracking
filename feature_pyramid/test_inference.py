import torch
from pytorch_lightning.cli import LightningCLI

from model import TransformerFCN


class TransformerFCNModule(TransformerFCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tensor = [tuple([
            torch.rand(16, 64, 32, 32),
            torch.rand(16, 96, 16, 16),
            torch.rand(16, 192, 8, 8),
        ])]
        # Save model to torchscript file and trace with example tensor
        super().to_torchscript(file_path='data/models/torch-script/transformer_fcn.pt',
                               method='trace',
                               example_inputs=tensor)


def cli_main():
    # Init model with YAML config via command line
    cli = LightningCLI(TransformerFCNModule, run=False)
    print(cli.model)


if __name__ == '__main__':
    cli_main()
