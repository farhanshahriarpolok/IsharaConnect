"""Unit tests for the Unified Predictor."""

import pytest
import numpy as np
from core_engine.inference.predictor import RealTimePredictor

@pytest.fixture
def predictor():
    # Since ONNX models might not be present during tests, we can just test the buffer logic
    # and ensure it doesn't crash on either 128D or 151D inputs.
    return RealTimePredictor(model_path="dummy.onnx", spatial_model_path="dummy_spatial.onnx")

def test_process_frame_single_hand(predictor):
    """Test predictor with 128D single-hand vector."""
    dummy_landmarks = np.zeros(128, dtype=np.float32)
    
    # Fill the sequence buffer
    for _ in range(predictor.sequence_length - 1):
        result = predictor.process_frame(dummy_landmarks)
        assert result is None
        
    # Final frame should trigger inference (or return None if model not loaded)
    result = predictor.process_frame(dummy_landmarks)
    # Model not loaded, so it should return None, but it shouldn't crash
    assert result is None
    assert len(predictor.landmark_buffer) == predictor.sequence_length
    assert len(predictor.spatial_buffer) == 0

def test_process_frame_dual_hand(predictor):
    """Test predictor with 151D dual-hand spatial vector."""
    dummy_landmarks = np.zeros(151, dtype=np.float32)
    
    # Fill the sequence buffer
    for _ in range(predictor.sequence_length - 1):
        result = predictor.process_frame(dummy_landmarks)
        assert result is None
        
    # Final frame should trigger inference (or return None if model not loaded)
    result = predictor.process_frame(dummy_landmarks)
    # Model not loaded, so it should return None, but it shouldn't crash
    assert result is None
    assert len(predictor.spatial_buffer) == predictor.sequence_length
    assert len(predictor.landmark_buffer) == 0

def test_invalid_shape(predictor):
    """Test predictor with invalid shape."""
    dummy_landmarks = np.zeros(100, dtype=np.float32)
    result = predictor.process_frame(dummy_landmarks)
    assert result is None
    assert len(predictor.landmark_buffer) == 0
    assert len(predictor.spatial_buffer) == 0
