"""Pytest configuration and shared fixtures for IsharaConnect."""

import numpy as np
import pytest


@pytest.fixture
def sample_hand_landmarks() -> np.ndarray:
    """Generate synthetic 21-landmark coordinates for a single hand."""
    # 21 points with 3 coordinates (x, y, z)
    np.random.seed(42)
    landmarks = np.random.uniform(low=0.2, high=0.8, size=(21, 3)).astype(np.float32)
    # Set wrist at specific location
    landmarks[0] = np.array([0.5, 0.7, 0.0], dtype=np.float32)
    # Set middle finger MCP
    landmarks[9] = np.array([0.5, 0.4, 0.0], dtype=np.float32)
    return landmarks
