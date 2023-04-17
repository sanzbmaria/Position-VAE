"""
PyTorch Lightning data module for VAEs. 

This module provides train, validation and test dataloaders for the freely moving NHP dataset .

Classes:
    - NHPDataset: A PyTorch dataset for NHP data.
    - VAEDataset: A PyTorch Lightning data module for VAEs.

Example:
    vae_data_module = VAEDataset(data_path='./my_dataset.pt', train_batch_size=32, val_batch_size=64) trainer.fit(model, vae_data_module) 

"""


import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class NHPDataset(torch.utils.data.Dataset):
    """
    Time series dataset

    Args:
        data (torch.Tensor): Time series data
    """

    def __init__(self, data) -> None:
        super().__init__()
        self.dataset = data

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):

        start = index * 1
        end = start + 1

        # block by block
        return self.dataset[start:end]


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module for VAEs

    Args:
        data_path (str): Path to the dataset
        train_batch_size (int, optional): Training batch size. Defaults to 8.
        val_batch_size (int, optional): Validation batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 24.
        pin_memory (bool, optional): Pin memory. Defaults to False.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 24,
        pin_memory: bool = False,
        type: str = "block",
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the dataset
        Args:
            stage (Optional[str], optional): Stage. Defaults to None.

        """
        dataset = torch.load(self.data_dir)
        # join the last  two dimensions
        dataset = dataset.reshape(dataset.shape[0], -1)

        # split the dataset
        train_data, test_data = train_test_split(dataset, shuffle=False)
        train_data, val_data = train_test_split(train_data, shuffle=False)

        # scale the data

        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # train_data = scaler.fit_transform(train_data)
        # test_data = scaler.transform(test_data)
        # val_data = scaler.transform(val_data)

        # convert to tensor
        # train_data = torch.from_numpy(train_data).float()
        # test_data = torch.from_numpy(test_data).float()
        # val_data = torch.from_numpy(val_data).float()

        # create the dataset
        self.train_dataset = NHPDataset(
            train_data)
        self.val_dataset = NHPDataset(
            val_data)
        self.test_dataset = NHPDataset(
            test_data)

    def train_dataloader(self):
        """
        Training dataloader
        Returns:
            DataLoader (torch.utils.data.DataLoader): Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Validation dataloader
        Returns:
            DataLoader (torch.utils.data.DataLoader): Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """
        Test dataloader
        Returns:
            DataLoader (torch.utils.data.DataLoader): Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
