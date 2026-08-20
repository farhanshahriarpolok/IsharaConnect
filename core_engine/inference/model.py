"""Neural Sequence Classifier for BdSL."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """Temporal Attention mechanism to weigh important frames in a sequence."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        attn_weights = self.attention(lstm_out) # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum across temporal dimension
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch_size, hidden_dim)
        return context_vector


class BdSLSequenceClassifier(nn.Module):
    """Bidirectional LSTM Classifier for continuous sequence gestures.
    
    Architecture:
    Input -> Bi-LSTM (2 layers) -> Temporal Attention -> FC Head -> Output
    """

    def __init__(
        self, 
        input_dim: int = 126, 
        hidden_dim: int = 128, 
        num_classes: int = 24, 
        num_layers: int = 2, 
        dropout: float = 0.3
    ):
        super().__init__()
        
        # We expect 128 features from normalizer, but 126 coordinate features if ignoring presence flags,
        # Let's support 128 to match LandmarkNormalizer.TOTAL_FEATURE_DIM
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Bi-LSTM doubles the hidden dimension
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, feature_dim)
               e.g., (B, 30, 128)
               
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Context vector after attention weighting over the sequence
        context = self.attention(lstm_out)
        
        # Classification head
        logits = self.classifier(context)
        return logits
