"""Bidirectional LSTM with Multi-Head Attention for BdSL Temporal ISLR Recognition.

Processes 60-frame 151-D spatial-temporal landmark sequences (B, 60, 151)
with temporal feature encoding, self-attention pooling, and dynamic sign classification.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism over temporal frame embeddings."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)


class BidirectionalLSTMAttention(nn.Module):
    """Deep Temporal ISLR Classifier: 2-layer BiLSTM + Self-Attention + Dropout."""

    def __init__(
        self,
        input_dim: int = 151,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 50,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Input feature projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM backbone
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Self-Attention pooling layer over 2 * hidden_dim = 256
        self.attention = TemporalSelfAttention(embed_dim=hidden_dim * 2, num_heads=4)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 60, 151)
        proj = self.input_proj(x)  # (B, 60, 128)
        lstm_out, _ = self.lstm(proj)  # (B, 60, 256)
        attn_out = self.attention(lstm_out)  # (B, 60, 256)

        # Global average pooling over time
        pooled = torch.mean(attn_out, dim=1)  # (B, 256)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits

    def export_onnx(self, output_path: str, sample_input: Optional[torch.Tensor] = None):
        """Exports the model to ONNX runtime format."""
        self.eval()
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if sample_input is None:
            sample_input = torch.randn(1, 60, self.input_dim, dtype=torch.float32)

        torch.onnx.export(
            self,
            sample_input,
            str(out_file),
            input_names=["temporal_landmarks"],
            output_names=["probabilities"],
            dynamic_axes={
                "temporal_landmarks": {0: "batch_size"},
                "probabilities": {0: "batch_size"}
            },
            opset_version=14,
            dynamo=False
        )
