import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader


class CustomDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 6,
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
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.transform = transform

    def setup(self, stage: str):
        if stage == "test":
            self.dataset_test = self.get_data_func(self.data_dir,
                                                   test=True,
                                                   transform=self.transform)
        if stage == "predict":
            self.dataset_predict = self.get_data_func(self.data_dir,
                                                      predict=True,
                                                      transform=self.transform)
        if stage == "fit":
            self.dataset_train = self.get_data_func(self.data_dir,
                                                    train=True,
                                                    transform=self.transform)

    def dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          # num_workers=self.num_workers,
                          pin_memory=True)

    def train_dataloader(self):
        return self.dataloader(self.dataset_train)

    def val_dataloader(self):
        return self.dataloader(self.dataset_val)

    def test_dataloader(self):
        return self.dataloader(self.dataset_test)

    def predict_dataloader(self):
        return self.dataloader(self.dataset_predict)
