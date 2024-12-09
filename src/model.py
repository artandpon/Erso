import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any

class LargeScaleModel(pl.LightningModule):
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [512, 256, 128],
                 learning_rate: float = 1e-3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) 