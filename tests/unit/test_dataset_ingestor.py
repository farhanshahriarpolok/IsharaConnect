"""Unit tests for Dataset Ingestion and 151-D Spatial Extraction Pipeline."""

from pathlib import Path
import numpy as np
import pytest

from dataset.tools.download_bdsl_dataset import BdSLDatasetIngestor
from dataset.tools.extract_151d_spatial_dataset import extract_151d_from_landmarks


def test_bdsl_dataset_ingestor_load_labels():
    """Verify BdSLDatasetIngestor loads all canonical labels."""
    ingestor = BdSLDatasetIngestor(samples_per_class=2)
    labels = ingestor.load_labels()
    assert len(labels) >= 60


def test_bdsl_dataset_ingestor_canonical_and_augmented_hands():
    """Verify canonical hand synthesis and geometric augmentation."""
    ingestor = BdSLDatasetIngestor(samples_per_class=2)
    r_hand = ingestor._generate_canonical_hand(is_right=True, is_dual=False, sign_id=1)
    assert r_hand.shape == (21, 3)

    l_aug, r_aug = ingestor.generate_augmented_sample(None, r_hand)
    assert l_aug is None
    assert r_aug.shape == (21, 3)
    assert not np.array_equal(r_hand, r_aug)


def test_extract_151d_from_landmarks():
    """Verify extract_151d_from_landmarks computes exactly 151 dimensions."""
    left = np.random.randn(21, 3).astype(np.float32)
    right = np.random.randn(21, 3).astype(np.float32)

    vec = extract_151d_from_landmarks(left, right)
    assert vec.shape == (151,)
    assert vec.dtype == np.float32

    # Test single-handed mode
    vec_single = extract_151d_from_landmarks(None, right)
    assert vec_single.shape == (151,)
