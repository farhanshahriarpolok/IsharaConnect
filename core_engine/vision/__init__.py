"""Vision subsystem: MediaPipe Hands tracking, camera capture, spatial features, and DTW matching."""

from core_engine.vision.hand_detector import HandDetector
from core_engine.vision.spatial_hand_engine import SpatialHandEngine
from core_engine.vision.dtw_matcher import DTWMotionMatcher

__all__ = ["HandDetector", "SpatialHandEngine", "DTWMotionMatcher"]
