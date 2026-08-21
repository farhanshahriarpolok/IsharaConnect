"""Unit tests for EnsemblePredictor."""

from unittest.mock import MagicMock
import pytest
import numpy as np

from core_engine.inference.ensemble_predictor import EnsemblePredictor


def _create_single_finger_lm():
    """Generates synthetic landmarks for 1 index finger extended."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.8, 0.0]
    # Index finger extended
    lm[5] = [0.5, 0.6, 0.0]
    lm[6] = [0.5, 0.45, 0.0]
    lm[7] = [0.5, 0.35, 0.0]
    lm[8] = [0.5, 0.20, 0.0]
    # Other fingers curled
    for mcp, tip in [(9, 12), (13, 16), (17, 20)]:
        lm[mcp] = [0.6, 0.6, 0.0]
        lm[tip] = [0.6, 0.68, 0.0]
    return lm


def test_ensemble_geometric_fast_path():
    """Unambiguous geometric sign should return immediately with source='geometric'."""
    mock_neural = MagicMock()
    predictor = EnsemblePredictor(neural_predictor=mock_neural, geometric_threshold=0.80)

    lm = _create_single_finger_lm()
    pred = predictor.predict(
        feature_vector=np.zeros(151, dtype=np.float32),
        left_landmarks=None,
        right_landmarks=lm
    )

    assert pred is not None
    assert pred["source"] == "geometric"
    assert "১" in pred["label_bn"]
    assert pred["confidence"] >= 0.80
    assert "finger_status" in pred
    # Neural model shouldn't be called if geometric is unambiguous
    assert mock_neural.process_frame.call_count == 0


def test_ensemble_sensitivity_modes():
    """Sensitivity level adjustments change thresholds as expected."""
    predictor = EnsemblePredictor()

    predictor.set_sensitivity("high")
    assert predictor.geometric_threshold == 0.60

    predictor.set_sensitivity("strict")
    assert predictor.geometric_threshold == 0.80

    predictor.set_sensitivity("normal")
    assert predictor.geometric_threshold == 0.70


def test_ensemble_dtw_integration():
    """Dynamic gesture temporal buffer routes through DTW engine."""
    predictor = EnsemblePredictor()
    ref_seq = predictor.dtw_matcher.generate_synthetic_reference("dhonnobad")

    pred = predictor.predict(
        feature_vector=None,
        left_landmarks=None,
        right_landmarks=None,
        temporal_buffer=ref_seq,
        target_sign="dhonnobad"
    )

    assert pred is not None
    assert pred["source"] == "dtw"
    assert "ধন্যবাদ" in pred["label_bn"]
    assert pred["confidence"] >= 0.80


def test_ensemble_neural_fallback():
    """When geometric is not detected, neural model output is returned with checklist attached."""
    mock_neural = MagicMock()
    mock_neural.process_frame.return_value = {
        "label_bn": "আমি",
        "label_en": "I / Me",
        "confidence": 0.88,
        "is_stable": True
    }

    predictor = EnsemblePredictor(neural_predictor=mock_neural)
    # Ambiguous landmark layout
    ambiguous_lm = np.ones((21, 3), dtype=np.float32) * 0.5

    pred = predictor.predict(
        feature_vector=np.ones(151, dtype=np.float32),
        left_landmarks=None,
        right_landmarks=ambiguous_lm
    )

    assert pred is not None
    assert pred["source"] == "neural"
    assert pred["label_bn"] == "আমি"
    assert "finger_status" in pred
    assert mock_neural.process_frame.call_count == 1
