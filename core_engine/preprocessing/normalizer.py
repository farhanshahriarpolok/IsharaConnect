"""Landmark Normalizer for IsharaConnect.

Transforms raw 3D MediaPipe hand landmarks (21 points per hand) into a
scale-, translation-, and position-invariant 128-dimensional feature vector.
"""

from typing import Any, List, Optional
import numpy as np


class LandmarkNormalizer:
    """Normalizes 21-hand landmark coordinates.

    Applies:
    1. Wrist origin translation: Landmark 0 becomes (0, 0, 0).
    2. Scale invariance: Divides coordinates by wrist-to-middle-MCP distance.
    3. Left/Right hand layout packing (126 coordinates + 2 presence flags = 128 features).
    """

    NUM_LANDMARKS_PER_HAND = 21
    COORDINATES_PER_LANDMARK = 3  # (x, y, z)
    FEATURES_PER_HAND = NUM_LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK  # 63
    TOTAL_FEATURE_DIM = (FEATURES_PER_HAND * 2) + 2  # 128 (Left hand + Right hand + 2 presence flags)

    @classmethod
    def normalize_single_hand(
        cls, landmarks: np.ndarray, epsilon: float = 1e-6
    ) -> np.ndarray:
        """Normalize a single hand's 21 landmarks (shape: [21, 3]).

        Args:
            landmarks: Array of shape (21, 3) representing (x, y, z) coordinates.
            epsilon: Small constant to avoid division by zero.

        Returns:
            Normalized 1D array of shape (63,).
        """
        if landmarks.shape != (cls.NUM_LANDMARKS_PER_HAND, cls.COORDINATES_PER_LANDMARK):
            raise ValueError(
                f"Expected landmarks array of shape (21, 3), got {landmarks.shape}"
            )

        # 1. Translate wrist (landmark 0) to origin (0, 0, 0)
        wrist = landmarks[0, :]
        translated = landmarks - wrist

        # 2. Scale invariance using distance from wrist (0) to middle finger MCP (9)
        middle_mcp = translated[9, :]
        hand_scale = float(np.linalg.norm(middle_mcp))
        hand_scale = max(hand_scale, epsilon)

        normalized = translated / hand_scale
        return normalized.flatten()

    @classmethod
    def process_frame(
        cls,
        left_hand_landmarks: Optional[np.ndarray] = None,
        right_hand_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Combine and normalize both hands into a standard 128-dim feature vector.

        Vector structure:
        - [0:63]   : Left hand normalized coordinates (or zeros if absent)
        - [63:126] : Right hand normalized coordinates (or zeros if absent)
        - [126]    : Left hand presence flag (1.0 if detected, 0.0 otherwise)
        - [127]    : Right hand presence flag (1.0 if detected, 0.0 otherwise)

        Returns:
            1D numpy array of shape (128,) with dtype float32.
        """
        vector = np.zeros(cls.TOTAL_FEATURE_DIM, dtype=np.float32)

        if left_hand_landmarks is not None:
            vector[0:63] = cls.normalize_single_hand(left_hand_landmarks)
            vector[126] = 1.0

        if right_hand_landmarks is not None:
            vector[63:126] = cls.normalize_single_hand(right_hand_landmarks)
            vector[127] = 1.0

        return vector
