"""Unit tests for LandmarkNormalizer."""

import numpy as np
import pytest
from core_engine.preprocessing.normalizer import LandmarkNormalizer


def test_single_hand_normalization(sample_hand_landmarks: np.ndarray) -> None:
    """Test wrist translation and scale invariance on a single hand."""
    normalized = LandmarkNormalizer.normalize_single_hand(sample_hand_landmarks)

    assert normalized.shape == (63,)
    # Landmark 0 (wrist) should be translated to (0, 0, 0)
    assert np.allclose(normalized[0:3], [0.0, 0.0, 0.0], atol=1e-5)

    # Test scale invariance: translating the original points should produce identical normalized result
    translated_landmarks = sample_hand_landmarks + np.array([10.0, -5.0, 2.0], dtype=np.float32)
    normalized_translated = LandmarkNormalizer.normalize_single_hand(translated_landmarks)
    assert np.allclose(normalized, normalized_translated, atol=1e-5)


def test_process_frame_dual_hands(sample_hand_landmarks: np.ndarray) -> None:
    """Test full 128-dimensional vector packaging."""
    left_hand = sample_hand_landmarks
    right_hand = sample_hand_landmarks + 0.1

    vector = LandmarkNormalizer.process_frame(left_hand, right_hand)
    assert vector.shape == (128,)
    assert vector[126] == 1.0  # Left presence
    assert vector[127] == 1.0  # Right presence


def test_process_frame_single_hand(sample_hand_landmarks: np.ndarray) -> None:
    """Test frame processing when only right hand is present."""
    vector = LandmarkNormalizer.process_frame(left_hand_landmarks=None, right_hand_landmarks=sample_hand_landmarks)
    assert vector.shape == (128,)
    assert vector[126] == 0.0  # Left presence is 0
    assert vector[127] == 1.0  # Right presence is 1
    assert np.all(vector[0:63] == 0.0)  # Left hand slice is zero
