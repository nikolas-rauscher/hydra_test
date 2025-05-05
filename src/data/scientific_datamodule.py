from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os


class ScientificPapersDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.files = self._get_all_parquet_files(data_path)
        self.data = self._load_data()
        
    def _get_all_parquet_files(self, directory):
        parquet_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        return parquet_files
        
    def _load_data(self):
        # Load all parquet files into a single dataframe
        data = []
        for file in self.files:
            df = pd.read_parquet(file)
            data.append(df)
        
        if len(data) > 0:
            return pd.concat(data, ignore_index=True)
        return pd.DataFrame()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Extract text and metadata
        text = item.get('Text', '')
        
        # Create sample dictionary
        sample = {
            'text': text,
            'title': item.get('title', ''),
            'authors': item.get('authors', ''),
            'publication_year': item.get('publication_year', 0),
            'journal': item.get('journal', ''),
            'doi': item.get('doi', ''),
            'id': item.get('ID', '')
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class ScientificPapersDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (80, 10, 10),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ScientificPapersDataset(self.hparams.data_dir)
            
            # Calculate split sizes
            dataset_size = len(dataset)
            train_size = int(dataset_size * (self.hparams.train_val_test_split[0]/100))
            val_size = int(dataset_size * (self.hparams.train_val_test_split[1]/100))
            test_size = dataset_size - train_size - val_size
            
            # Split the dataset
            self.data_train, self.data_val, self.data_test = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        ) 