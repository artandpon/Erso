import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import pandas as pd
import numpy as np

class LargeScaleDataset(Dataset):
    def __init__(self, 
                 file_path: str,
                 batch_size: int = 1024,
                 chunk_size: int = 100000):
        """
        Dataset class optimized for large scale data processing
        
        Args:
            file_path: Path to the data file (CSV format)
            batch_size: Size of batches for training
            chunk_size: Number of rows to load at once
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # Get total number of rows without loading entire dataset
        self.total_rows = sum(1 for _ in pd.read_csv(file_path, chunksize=chunk_size))
        
    def __len__(self) -> int:
        return self.total_rows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate which chunk contains our index
        chunk_idx = idx // self.chunk_size
        row_idx = idx % self.chunk_size
        
        # Load only the required chunk
        chunk = pd.read_csv(self.file_path, 
                          skiprows=chunk_idx * self.chunk_size,
                          nrows=self.chunk_size)
        
        # Get the specific row
        row = chunk.iloc[row_idx]
        
        # Assuming last column is target, rest are features
        features = torch.FloatTensor(row[:-1].values)
        target = torch.FloatTensor([row[-1]])
        
        return features, target

def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 1024,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for training and validation
    """
    train_dataset = LargeScaleDataset(train_path, batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_path:
        val_dataset = LargeScaleDataset(val_path, batch_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader 