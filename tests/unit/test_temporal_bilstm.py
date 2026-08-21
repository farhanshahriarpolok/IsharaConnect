"""Unit tests for Temporal BiLSTM + Attention Model."""

import numpy as np
import pytest
import torch

from core_engine.inference.temporal_bilstm import (
    BidirectionalLSTMAttention,
    TemporalSelfAttention,
)


def test_temporal_self_attention_shape():
    """Self-attention preserves temporal batch and feature dimensions."""
    attn = TemporalSelfAttention(embed_dim=256, num_heads=4)
    x = torch.randn(2, 60, 256)
    out = attn(x)
    assert out.shape == (2, 60, 256)


def test_bilstm_attention_forward():
    """Model forward pass maps (B, 60, 151) input to (B, num_classes) logits."""
    model = BidirectionalLSTMAttention(
        input_dim=151,
        hidden_dim=64,
        num_layers=2,
        num_classes=10,
        dropout=0.2
    )
    x = torch.randn(4, 60, 151)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_bilstm_onnx_export(tmp_path):
    """Model exports to valid ONNX file format."""
    model = BidirectionalLSTMAttention(
        input_dim=151,
        hidden_dim=32,
        num_layers=1,
        num_classes=5
    )
    onnx_file = tmp_path / "test_temporal.onnx"
    model.export_onnx(str(onnx_file))
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 1000
