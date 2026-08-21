"""Unit tests for Dynamic Time Warping (DTW) Motion & Gesture Matcher."""

import pytest
import numpy as np
from core_engine.vision.dtw_matcher import DTWMotionMatcher


def test_dtw_distance_identical_sequences():
    """Identical sequences must have zero DTW distance and 100% match score."""
    matcher = DTWMotionMatcher()
    seq = np.random.randn(30, 151).astype(np.float32)

    total_cost, norm_cost, path, matrix = matcher.compute_dtw_distance(seq, seq)
    assert pytest.approx(total_cost, abs=1e-4) == 0.0
    assert pytest.approx(norm_cost, abs=1e-4) == 0.0
    assert len(path) == 30

    eval_result = matcher.evaluate_gesture_accuracy(seq, seq)
    assert eval_result["distance"] == 0.0
    assert eval_result["score"] == 100.0
    assert eval_result["match_percentage"] == 100.0
    assert eval_result["is_match"] is True


def test_dtw_time_warped_sequences():
    """DTW should successfully align time-stretched sequences with high score."""
    matcher = DTWMotionMatcher(distance_scale=10.0, match_threshold=15.0)

    # Base trajectory: smooth sine wave
    t1 = np.linspace(0, np.pi, 20)
    seq1 = np.zeros((20, 151), dtype=np.float32)
    for i in range(20):
        seq1[i, :10] = np.sin(t1[i])

    # Time-stretched sequence: 35 frames of the same trajectory
    t2 = np.linspace(0, np.pi, 35)
    seq2 = np.zeros((35, 151), dtype=np.float32)
    for i in range(35):
        seq2[i, :10] = np.sin(t2[i])

    eval_result = matcher.evaluate_gesture_accuracy(seq1, seq2)
    assert eval_result["score"] >= 80.0
    assert eval_result["is_match"] is True
    assert eval_result["path_length"] >= 35


def test_dtw_dissimilar_sequences():
    """Completely opposite / orthogonal sequences should yield low scores."""
    matcher = DTWMotionMatcher(distance_scale=5.0, match_threshold=5.0)

    seq1 = np.ones((30, 151), dtype=np.float32) * 5.0
    seq2 = np.ones((30, 151), dtype=np.float32) * -5.0

    eval_result = matcher.evaluate_gesture_accuracy(seq1, seq2)
    assert eval_result["score"] < 40.0
    assert eval_result["is_match"] is False
    assert eval_result["normalized_distance"] > 10.0


def test_evaluate_gesture_accuracy_schema():
    """Verify dictionary structure and type constraints."""
    matcher = DTWMotionMatcher()
    seq1 = np.zeros((15, 151), dtype=np.float32)
    seq2 = np.zeros((20, 151), dtype=np.float32)

    res = matcher.evaluate_gesture_accuracy(seq1, seq2)
    assert "distance" in res
    assert "normalized_distance" in res
    assert "score" in res
    assert "match_percentage" in res
    assert "is_match" in res
    assert "path_length" in res
    assert "warping_path" in res
    assert isinstance(res["warping_path"], list)
    assert 0.0 <= res["score"] <= 100.0


def test_match_sign_classification():
    """Verify candidate classification identifies the closest reference sign."""
    matcher = DTWMotionMatcher()

    ref_dhonnobad = matcher.generate_synthetic_reference("dhonnobad")
    ref_kemon = matcher.generate_synthetic_reference("kemon_achen")
    ref_sahajjo = matcher.generate_synthetic_reference("sahajjo")

    candidates = {
        "dhonnobad": ref_dhonnobad,
        "kemon_achen": ref_kemon,
        "sahajjo": ref_sahajjo
    }

    # Test with noisy dhonnobad
    noisy_dhonnobad = ref_dhonnobad + np.random.normal(0, 0.05, ref_dhonnobad.shape).astype(np.float32)
    match_result = matcher.match_sign(noisy_dhonnobad, candidates=candidates)

    assert match_result["best_match"] == "dhonnobad"
    assert match_result["best_score"] >= 80.0
    assert match_result["is_match"] is True


def test_empty_sequence_safety():
    """Empty inputs should be handled gracefully without raising exceptions."""
    matcher = DTWMotionMatcher()
    empty = np.array([], dtype=np.float32)
    valid = np.zeros((30, 151), dtype=np.float32)

    res1 = matcher.evaluate_gesture_accuracy(empty, valid)
    assert res1["score"] == 0.0
    assert res1["is_match"] is False

    res2 = matcher.evaluate_gesture_accuracy(valid, empty)
    assert res2["score"] == 0.0
    assert res2["is_match"] is False
