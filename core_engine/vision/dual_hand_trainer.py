"""PyTorch Models and Datasets for Dual-Hand 151D Spatial Features."""

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)

class SpatialFeatureDataset(Dataset):
    """Dataset for loading pre-extracted 151D Spatial Features.
    
    Expected input format for each sample:
    - 42 normalized landmarks (42 * 3 = 126 dims)
    - 5x5 touch matrix (25 dims)
    Total: 151 dims
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: NumPy array of shape (N, seq_len, 151) or (N, 151)
            labels: NumPy array of shape (N,) containing class indices
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ResidualBlock(nn.Module):
    """Simple 1D Residual Block for MLP."""
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DualHandSpatialModel(nn.Module):
    """Deep Residual MLP tailored for 151-D spatial vectors."""
    def __init__(self, input_dim: int = 151, hidden_dim: int = 256, num_classes: int = 63):
        super().__init__()
        
        # We expect a static frame vector (N, 151) or temporal collapsed to (N, 151)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Ensure correct shape (batch, features)
        if x.dim() > 2:
            # If temporal sequence is passed, pool it across time (mean pooling)
            x = torch.mean(x, dim=1)
            
        out = self.input_layer(x)
        out = self.res_blocks(out)
        out = self.classifier(out)
        return out
