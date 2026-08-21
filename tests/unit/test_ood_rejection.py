import pytest
import numpy as np
from core_engine.inference.predictor import RealTimePredictor
from core_engine.inference.config import InferenceConfig
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_ort_session():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    return session

@patch('core_engine.inference.predictor.Path.exists')
@patch('core_engine.inference.predictor.ort', create=True)
def test_ood_rejection_high_entropy(mock_ort, mock_exists, mock_ort_session):
    mock_exists.return_value = True
    mock_ort.InferenceSession.return_value = mock_ort_session
    
    config = InferenceConfig(
        sequence_length=1, 
        allow_ood_rejection=True,
        entropy_threshold=0.65,
        confidence_threshold=0.8
    )
    predictor = RealTimePredictor(config=config, model_path="dummy.onnx", agreement_threshold=0.0)
    
    # Mock softmax probabilities to be uniform (high entropy)
    # logits need to be identical
    mock_ort_session.run.return_value = [np.array([[0.333, 0.333, 0.334]])]
    
    dummy_landmarks = np.zeros(128, dtype=np.float32)
    result = predictor.process_frame(dummy_landmarks)
    
    # Check if prediction buffer got -1 (Unknown)
    assert predictor.prediction_buffer[-1] == -1
    
    # And it should emit Unknown Sign if agreement is met
    # agreement_window is 10, so let's push 9 more frames
    for _ in range(9):
        result = predictor.process_frame(dummy_landmarks)
        
    assert result is not None
    assert result["sign_id"] == -1
    assert result["label_en"] == "Unknown Sign"

@patch('core_engine.inference.predictor.Path.exists')
@patch('core_engine.inference.predictor.ort', create=True)
def test_ood_acceptance_low_entropy(mock_ort, mock_exists, mock_ort_session):
    mock_exists.return_value = True
    mock_ort.InferenceSession.return_value = mock_ort_session
    
    config = InferenceConfig(
        sequence_length=1, 
        allow_ood_rejection=True,
        entropy_threshold=0.65,
        confidence_threshold=0.8
    )
    predictor = RealTimePredictor(config=config, model_path="dummy.onnx", agreement_threshold=0.0)
    
    # Mock logits for high confidence (low entropy)
    # [10.0, 0.0, 0.0] -> ~[1.0, 0.0, 0.0] prob
    mock_ort_session.run.return_value = [np.array([[10.0, 0.0, 0.0]])]
    
    dummy_landmarks = np.zeros(128, dtype=np.float32)
    predictor.labels = {0: {"bangla": "Zero", "english": "Zero"}}
    
    for i in range(10):
        result = predictor.process_frame(dummy_landmarks)
        
    assert result is not None
    assert result["sign_id"] == 0
    assert result["label_en"] == "Zero"
