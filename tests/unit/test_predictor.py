"""Unit tests for RealTimePredictor temporal smoothing and debounce."""

import numpy as np
import pytest

from core_engine.inference.predictor import RealTimePredictor


class MockOrtSession:
    """Mock ONNX Runtime session."""
    def __init__(self, num_classes=24, force_class=-1, force_confidence=0.9):
        self.num_classes = num_classes
        self.force_class = force_class
        self.force_confidence = force_confidence

    class MockInput:
        def __init__(self, name):
            self.name = name

    def get_inputs(self):
        return [self.MockInput("input")]

    def run(self, output_names, input_feed):
        # input_feed["input"] shape (1, 30, 128)
        logits = np.zeros((1, self.num_classes), dtype=np.float32)
        if self.force_class != -1:
            logits[0, self.force_class] = 10.0 # High value for softmax
        else:
            logits[0, :] = 1.0 # Uniform
            
        return [logits]


@pytest.fixture
def mock_predictor(monkeypatch):
    """Predictor with mocked ONNX runtime."""
    predictor = RealTimePredictor(
        model_path="dummy_path.onnx", # Won't actually load due to mock
        sequence_length=5, # Shortened for testing
        agreement_window=3,
        debounce_cooldown_sec=1.0
    )
    
    # Inject mock session
    predictor.ort_session = MockOrtSession(force_class=5)
    predictor.input_name = "input"
    
    # Mock labels
    predictor.labels = {5: {"bangla": "টেস্ট", "english": "Test"}}
    
    return predictor


def test_buffer_management(mock_predictor):
    """Test that predictor waits for sequence_length frames."""
    dummy_landmark = np.zeros((128,), dtype=np.float32)
    
    # Push 4 frames (sequence_length is 5)
    for _ in range(4):
        res = mock_predictor.process_frame(dummy_landmark)
        assert res is None
        
    # 5th frame should trigger inference
    res = mock_predictor.process_frame(dummy_landmark)
    # But agreement window is 3, so still None
    assert res is None
    
    # 6th frame
    res = mock_predictor.process_frame(dummy_landmark)
    assert res is None
    
    # 7th frame -> agreement window (3) is full
    res = mock_predictor.process_frame(dummy_landmark)
    assert res is not None
    assert res["sign_id"] == 5
    assert res["is_stable"] is True


def test_debounce_cooldown(mock_predictor):
    """Test that the same class isn't emitted rapidly."""
    dummy_landmark = np.zeros((128,), dtype=np.float32)
    
    # Fill buffers to trigger first emission
    for _ in range(7):
        res = mock_predictor.process_frame(dummy_landmark)
        
    assert res is not None
    assert res["sign_id"] == 5
    
    # Next frame immediately after should be debounced (return None)
    res2 = mock_predictor.process_frame(dummy_landmark)
    assert res2 is None
    
    # Simulate time passing > 1.0s
    mock_predictor.last_emitted_time -= 1.1
    
    # Should emit again
    res3 = mock_predictor.process_frame(dummy_landmark)
    assert res3 is not None
    assert res3["sign_id"] == 5
