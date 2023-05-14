import pytorch_lightning as pl
from torch import optim

class HandKeypointDetector(pl.LightningModule):
    def __init__(self, backbone, head, config):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.config = config

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

