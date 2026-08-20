"""Unit tests for HandDetector vision layer."""

import numpy as np
import pytest

from core_engine.vision.hand_detector import HandDetector
from core_engine.preprocessing.normalizer import LandmarkNormalizer


def test_hand_detector_initialization() -> None:
    """Test that HandDetector initializes without errors."""
    detector = HandDetector(static_image_mode=True, max_num_hands=2)
    assert detector.hands is not None
    detector.close()


def test_extract_landmarks_no_hands() -> None:
    """Test extraction on an image with no hands (empty state)."""
    detector = HandDetector(static_image_mode=True, max_num_hands=2)
    
    # We simulate the empty state without actually running process() on a real image
    # by simply calling extract_landmarks when last_results is None.
    detector.last_results = None
    extraction = detector.extract_landmarks((480, 640, 3))
    
    assert extraction["hands_detected"] == 0
    assert extraction["raw_left"] is None
    assert extraction["raw_right"] is None
    assert extraction["handedness"] == []
    
    # Verify integration with Normalizer handles None properly
    feature_vector = LandmarkNormalizer.process_frame(
        extraction["raw_left"], extraction["raw_right"]
    )
    
    assert feature_vector.shape == (128,)
    assert np.all(feature_vector == 0.0)
    
    detector.close()


def test_extract_landmarks_single_hand(sample_hand_landmarks: np.ndarray) -> None:
    """Test extraction logic by mocking MediaPipe results for a single Right hand."""
    detector = HandDetector(static_image_mode=True, max_num_hands=2)
    
    # Mocking MediaPipe output classes
    class MockLabel:
        def __init__(self, label: str):
            self.label = label
            
    class MockClassification:
        def __init__(self, label: str):
            self.classification = [MockLabel(label)]

    class MockLandmark:
        def __init__(self, x: float, y: float, z: float):
            self.x = x
            self.y = y
            self.z = z

    class MockHandLandmarks:
        def __init__(self, coords: np.ndarray):
            self.landmark = [MockLandmark(float(c[0]), float(c[1]), float(c[2])) for c in coords]

    class MockResults:
        def __init__(self, multi_handedness, multi_hand_landmarks):
            self.multi_handedness = multi_handedness
            self.multi_hand_landmarks = multi_hand_landmarks

    # Set up mock results for a single Right hand
    detector.last_results = MockResults(
        multi_handedness=[MockClassification("Right")],
        multi_hand_landmarks=[MockHandLandmarks(sample_hand_landmarks)]
    )
    
    extraction = detector.extract_landmarks((480, 640, 3))
    
    assert extraction["hands_detected"] == 1
    assert extraction["handedness"] == ["Right"]
    assert extraction["raw_left"] is None
    assert extraction["raw_right"] is not None
    assert extraction["raw_right"].shape == (21, 3)
    
    # Check normalized values
    feature_vector = LandmarkNormalizer.process_frame(
        extraction["raw_left"], extraction["raw_right"]
    )
    
    assert feature_vector.shape == (128,)
    assert feature_vector[126] == 0.0  # Left not present
    assert feature_vector[127] == 1.0  # Right present
    assert np.all(feature_vector[0:63] == 0.0) # Left padded with zeros
    
    detector.close()
