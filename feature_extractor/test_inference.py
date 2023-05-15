import torch
from pytorch_lightning.cli import LightningCLI

from feature_extractor.model import MogaNet


class MogaNetModule(MogaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tensor = torch.rand(16, 3, 256, 256)
        # Save model to torchscript file and trace with example tensor
        # self.to_torchscript(file_path='data/models/torch-script/moganet.pt',
        #                     method='trace',
        #                     example_inputs=tensor)
        print(super().forward(tensor))



def cli_main():
    # Init model with YAML config via command line
    cli = LightningCLI(MogaNetModule, run=False)


if __name__ == '__main__':
    cli_main()
