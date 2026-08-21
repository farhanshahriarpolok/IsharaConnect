"""Unit tests for the Mobile TFLite Exporter."""

import os
import pytest
from scripts.export_tflite import export_to_tflite, build_tflite_flatbuffer


def test_build_tflite_flatbuffer():
    """Test building synthetic TFLite FlatBuffer bytes."""
    import numpy as np
    dummy_weights = {
        "fc1.weight": np.random.randn(64, 151).astype(np.float32),
        "fc1.bias": np.random.randn(64).astype(np.float32)
    }
    fb_bytes = build_tflite_flatbuffer(
        weights_dict=dummy_weights,
        input_shape=(1, 151),
        output_shape=(1, 63),
        quantize_mode="fp16"
    )

    assert isinstance(fb_bytes, bytes)
    assert len(fb_bytes) > 0
    assert b"TFL3" in fb_bytes


def test_export_spatial_model_tflite(tmp_path):
    """Test exporting 151-D spatial model to TFLite with size assertion <= 5MB."""
    out_path = str(tmp_path / "test_spatial_quant.tflite")
    result_path = export_to_tflite(
        model_type="spatial",
        checkpoint_path=None,
        output_path=out_path,
        input_dim=151,
        num_classes=63,
        quantize="fp16",
        max_size_mb=5.0
    )

    assert os.path.exists(result_path)
    file_size_mb = os.path.getsize(result_path) / (1024 * 1024)
    assert file_size_mb <= 5.0
    assert result_path.endswith(".tflite")


def test_export_sequence_model_tflite(tmp_path):
    """Test exporting 128-D sequence model to TFLite with int8 quantization."""
    out_path = str(tmp_path / "test_sequence_quant.tflite")
    result_path = export_to_tflite(
        model_type="sequence",
        checkpoint_path=None,
        output_path=out_path,
        input_dim=128,
        num_classes=24,
        quantize="int8",
        max_size_mb=5.0
    )

    assert os.path.exists(result_path)
    file_size_mb = os.path.getsize(result_path) / (1024 * 1024)
    assert file_size_mb <= 5.0
