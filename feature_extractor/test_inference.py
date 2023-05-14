import torch
from pytorch_lightning.cli import LightningCLI

from feature_extractor import MogaNet


class MogaNetModule(MogaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def cli_main():
    # Init model with YAML config via command line
    cli = LightningCLI(MogaNetModule, run=False)
    print(cli.model)
    # Test model on random data
    print([t.shape for t in cli.model(torch.rand(16, 3, 256, 256))])
    # Save model to torchscript file
    cli.model.to_torchscript('data/models/torch-script/moganet.pt')


if __name__ == '__main__':
    cli_main()
