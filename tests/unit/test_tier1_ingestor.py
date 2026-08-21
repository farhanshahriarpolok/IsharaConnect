"""Unit tests for Tier 1: BdSL Fingerspelling Ingestion Pipeline."""

import numpy as np
import pytest
from pathlib import Path

from dataset.ingestors.tier1_fingerspelling_ingestor import Tier1FingerspellingIngestor


def test_tier1_manifest_loading():
    """Tier 1 ingestor loads standard manifest with 49 classes."""
    ingestor = Tier1FingerspellingIngestor()
    assert ingestor.manifest_data is not None
    assert ingestor.manifest_data.get("feature_dimension") == 151
    assert len(ingestor.manifest_data.get("classes", [])) >= 45


def test_tier1_generate_mock_and_validate(tmp_path):
    """Tier 1 mock dataset generation produces valid 151-D spatial tensors."""
    ingestor = Tier1FingerspellingIngestor()
    out_dir = tmp_path / "tier1_test"
    X, y, save_path = ingestor.generate_mock_dataset(num_samples_per_class=5, output_dir=str(out_dir))

    assert X.shape[1] == 151
    assert len(X) == len(y)
    assert not np.isnan(X).any()

    # Validate generated dataset
    val = ingestor.validate(save_path)
    assert val["valid"] is True
    assert val["num_samples"] == len(X)
    assert val["feature_dim"] == 151
    assert val["has_nans"] is False

    # Check statistics
    stats = ingestor.get_statistics(save_path)
    assert stats["samples_count"] == len(X)
    assert stats["feature_dimension"] == 151


def test_tier1_image_extraction_fallback():
    """Empty or None image returns None safely."""
    ingestor = Tier1FingerspellingIngestor()
    vec = ingestor.extract_from_image(None)
    assert vec is None
