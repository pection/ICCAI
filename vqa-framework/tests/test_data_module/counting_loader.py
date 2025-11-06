from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl
from typing import Optional
import os
from pytorch_lightning import LightningDataModule


class CountingDataset(Dataset):
    """
    Simple dataset of just incrementing numbers.
    """

    def __init__(self, dim: int, length: int, start: int = 0):
        super().__init__()
        self.len = length
        self.data = start + torch.Tensor([range(length)] * dim).transpose(1, 0)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class CountingDataModule(LightningDataModule):
    def __init__(
        self,
        shuffle_train: bool = False,
        batch_size: int = 1,
        dm_workers: int = 0,
        **kwargs
    ):
        super().__init__()
        self.shuffle_train = shuffle_train
        self.batch_size = batch_size

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.workers = dm_workers

        self.save_hyperparameters(ignore=["workers"])

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_data = CountingDataset(dim=1, length=5000, start=0)
        self.test_data = CountingDataset(dim=1, length=1000, start=10000)
        self.val_data = CountingDataset(dim=1, length=500, start=6000)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.workers
        )


class CrashOnDemandModel(pl.LightningModule):
    """
    Crash on the CRASH_VAL th batch of data, where CRASH_VAL is an env variable
    """

    def __init__(
        self,
        logging=False,
        sync_dist: bool = True,
        dummy_arg: Optional[int] = None,
        weight_init: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.crash_val = int(os.environ["CRASH_VAL"])
        self.processed_batches = 0
        self.processed_batches_param = torch.nn.Parameter(
            torch.LongTensor([0]), requires_grad=False
        )
        self.logging = logging
        self.sync_dist = sync_dist
        self.save_hyperparameters(ignore=["logging", "sync_dist", "weight_init"])

        if weight_init is not None:
            # Initialize to constant
            self.linear.weight.data.fill_(weight_init)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, x):
        self.crash_val = int(os.environ["CRASH_VAL"])
        y = self(x)

        """
        for tmp in x.cpu().numpy().tolist():
            if tmp == self.crash_val:
                raise ZeroDivisionError
        """
        if self.processed_batches == self.crash_val:
            raise ZeroDivisionError

        loss = torch.min(x.detach().clone()).item() + torch.sum((y - x) ** 2)
        self.processed_batches += 1
        self.processed_batches_param += 1

        if self.logging:
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                sync_dist=self.sync_dist,
            )
            self.log(
                "batches_processed",
                self.processed_batches,
                on_step=True,
                on_epoch=True,
                sync_dist=self.sync_dist,
            )

        return loss

    def validation_step(self, x, batch_idx):
        y = self(x)
        loss = torch.sum((y - x) ** 2)
        if self.logging:
            self.log(
                "val_loss", loss, on_step=True, on_epoch=True, sync_dist=self.sync_dist
            )
        return loss

    def test_step(self, x):
        return 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
