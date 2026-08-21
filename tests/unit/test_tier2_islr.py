"""Unit tests for Tier 2: BdSL Isolated Sign Language Recognition (ISLR) Pipeline."""

import numpy as np
import pytest
from pathlib import Path

from dataset.ingestors.tier2_islr_ingestor import Tier2ISLRIngestor


def test_tier2_manifest_loading():
    """Tier 2 ingestor loads standard ISLR manifest."""
    ingestor = Tier2ISLRIngestor()
    assert ingestor.manifest_data is not None
    assert ingestor.manifest_data.get("sequence_length") == 60
    assert ingestor.manifest_data.get("feature_dimension") == 151


def test_tier2_sequence_length_normalization():
    """Resamples arbitrary length sequences (e.g. 45 or 80 frames) to exactly 60 frames."""
    ingestor = Tier2ISLRIngestor()
    
    # Under-length sequence (40 frames)
    raw_short = np.random.randn(40, 151).astype(np.float32)
    norm_short = ingestor.normalize_sequence_length(raw_short, target_length=60)
    assert norm_short.shape == (60, 151)

    # Over-length sequence (85 frames)
    raw_long = np.random.randn(85, 151).astype(np.float32)
    norm_long = ingestor.normalize_sequence_length(raw_long, target_length=60)
    assert norm_long.shape == (60, 151)


def test_tier2_generate_mock_and_validate(tmp_path):
    """Generates synthetic ISLR dataset with 60-frame sequences and validates schema."""
    ingestor = Tier2ISLRIngestor()
    out_dir = tmp_path / "tier2_test"
    X, y, save_path = ingestor.generate_mock_dataset(num_samples_per_class=3, num_classes=10, output_dir=str(out_dir))

    assert len(X.shape) == 3
    assert X.shape[1] == 60
    assert X.shape[2] == 151
    assert len(X) == len(y)

    val = ingestor.validate(save_path)
    assert val["valid"] is True
    assert val["sequence_length"] == 60
    assert val["feature_dim"] == 151

    stats = ingestor.get_statistics(save_path)
    assert stats["sequences_count"] == len(X)
    assert stats["sequence_frames"] == 60
