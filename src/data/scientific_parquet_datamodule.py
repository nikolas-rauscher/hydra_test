from typing import Any, Dict, Optional, Tuple, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import duckdb
import os
from glob import glob
import logging

class ScientificParquetDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        sample_size: int = None,
        max_docs: int = None,
        transform=None
    ):
        self.data_path = data_path
        self.transform = transform
        self.sample_size = sample_size
        self.max_docs = max_docs
        self.data = self._load_data()
        
    def _load_data(self):
        """Load text data from Parquet files"""
        logging.info(f"Loading data from {self.data_path}")
        
        # Ensure the path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Find all parquet files
        parquet_files = glob(os.path.join(self.data_path, "**/*.parquet"), recursive=True)
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in: {self.data_path}")
            
        logging.info(f"Found {len(parquet_files)} parquet files")
        
        # Build the query with sample size if provided
        sample_clause = ""
        if self.sample_size:
            sample_clause = f"USING SAMPLE {self.sample_size} ROWS"
            
        # Try to extract title and other metadata from parquet files
        try:
            query = f"""
                SELECT 
                    json_extract(metadata, '$.title') as title,
                    json_extract(metadata, '$.doi') as doi,
                    json_extract(metadata, '$.authors') as authors,
                    COALESCE("Text", text) as text
                FROM read_parquet('{self.data_path}/*.parquet')
                WHERE (metadata IS NOT NULL AND json_extract(metadata, '$.title') IS NOT NULL)
                OR "Text" IS NOT NULL OR text IS NOT NULL
                {sample_clause}
            """
            
            df = duckdb.sql(query).df()
            
        except Exception as e:
            logging.warning(f"Failed to parse with metadata structure: {str(e)}")
            logging.info("Falling back to direct parquet reading")
            
            # Fallback: try to read the parquet files directly
            dfs = []
            for file in parquet_files[:10]:  # Limit to first 10 files for testing
                try:
                    df_part = pd.read_parquet(file)
                    if 'Text' in df_part.columns:
                        df_part = df_part.rename(columns={'Text': 'text'})
                    dfs.append(df_part)
                except Exception as e:
                    logging.warning(f"Error reading {file}: {str(e)}")
                    
            if not dfs:
                raise ValueError("Could not read any parquet files")
                
            df = pd.concat(dfs, ignore_index=True)
            
            # Limit to max_docs if specified
            if self.max_docs and len(df) > self.max_docs:
                df = df.sample(self.max_docs, random_state=42)
        
        logging.info(f"Loaded {len(df):,} documents")
        return df
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Extract title and text
        title = item.get('title', '')
        text = item.get('text', '')
        
        if not isinstance(title, str):
            title = str(title) if title is not None else ""
            
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Create sample dictionary
        sample = {
            'text': text,
            'title': title,
            'texts': [title] if title else [text[:200]],  # For topic modeling, use titles or first part of text
            'authors': item.get('authors', ''),
            'doi': item.get('doi', '')
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class ScientificParquetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (80, 10, 10),
        batch_size: int = 32,
        sample_size: int = None,
        max_docs: int = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data and set variables: `self.data_train`, `self.data_val`, `self.data_test`"""
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ScientificParquetDataset(
                self.hparams.data_dir,
                sample_size=self.hparams.sample_size,
                max_docs=self.hparams.max_docs
            )
            
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
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        """Custom collate function for text batches"""
        titles = []
        texts = []
        texts_for_topic = []
        
        for item in batch:
            titles.append(item['title'])
            texts.append(item['text'])
            texts_for_topic.extend(item['texts'])
            
        return {
            'titles': titles,
            'texts': texts,
            'texts': texts_for_topic  # For topic modeling
        } 