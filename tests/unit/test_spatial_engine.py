import pytest
import numpy as np
import cv2
from core_engine.vision.spatial_hand_engine import SpatialHandEngine

@pytest.fixture
def spatial_engine():
    # Use static image mode for easier testing
    return SpatialHandEngine(static_image_mode=True)

def test_spatial_engine_initialization(spatial_engine):
    assert spatial_engine.hands is not None
    assert len(spatial_engine.fingertips) == 5

def test_spatial_features_extraction(spatial_engine):
    # Create a dummy image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    features = spatial_engine.extract_spatial_features(frame)
    
    assert "has_left" in features
    assert "has_right" in features
    assert "raw_landmarks" in features
    assert "normalized_landmarks" in features
    assert "touch_matrix" in features
    assert "orientation" in features
    
    assert features["raw_landmarks"].shape == (42, 3)
    assert features["normalized_landmarks"].shape == (42, 3)
    assert features["touch_matrix"].shape == (5, 5)
    
    assert "left" in features["orientation"]
    assert "right" in features["orientation"]
