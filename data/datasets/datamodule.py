import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader


class CustomDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 get_data_func=None,
                 train_ratio=0.75,
                 transform=None):
        super().__init__()
        self.dataset_val = None
        self.dataset_train = None
        self.dataset_predict = None
        self.dataset_test = None
        self.data_dir = data_dir
        self.get_data_func = get_data_func
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.transform = transform

    def setup(self, stage: str):
        if stage == "test.yaml":
            self.dataset_test = self.get_data_func(self.data_dir,
                                                   test=True)
        if stage == "predict":
            self.dataset_predict = self.get_data_func(self.data_dir,
                                                      predict=True)
        if stage == "fit":
            dataset_full = self.get_data_func(self.data_dir,
                                              train=True,
                                              val=True,
                                              transform=self.transform)
            train, val = int(len(dataset_full) * (1 - self.train_ratio)), int(
                len(dataset_full) * self.train_ratio)
            self.dataset_train, self.dataset_val = random_split(dataset_full, [train, val])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=self.batch_size)
