"""Hand Landmark Detection Engine using MediaPipe.

Provides the `HandDetector` class to extract 21 3D landmarks for up to 2 hands,
and packages them into a uniform format compatible with the normalizer.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions.hands as mp_hands
        import mediapipe.solutions.drawing_utils as mp_drawing
        import mediapipe.solutions.drawing_styles as mp_drawing_styles
    except (ImportError, AttributeError):
        mp_hands = None
        mp_drawing = None
        mp_drawing_styles = None

logger = logging.getLogger(__name__)


class HandDetector:
    """Detects hand landmarks using MediaPipe Hands."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the MediaPipe Hands detector."""
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        self.hands = None
        if self.mp_hands:
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            logger.warning("MediaPipe solutions are unavailable. HandDetector will run in passthrough mode.")

        # Cache last results for potential smoothing or external queries
        self.last_results = None

    def find_hands(self, image: np.ndarray, draw: bool = True) -> np.ndarray:
        """Process image, extract landmarks, and optionally draw them.

        Args:
            image: BGR image (e.g., from cv2.VideoCapture).
            draw: If True, draw landmarks and connections on the image.

        Returns:
            Annotated image (if draw is True) or original image (if False).
        """
        if not self.hands:
            return image.copy()
            
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference
        image_rgb.flags.writeable = False
        self.last_results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        annotated_image = image.copy()

        if draw and self.last_results.multi_hand_landmarks:
            for hand_landmarks in self.last_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return annotated_image

    def extract_landmarks(self, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Extract landmarks from the last processed frame.

        Returns a structured dictionary with structured left/right hand landmarks,
        handling cases with 0, 1, or 2 hands. Pads missing hands with None.

        Args:
            image_shape: Shape of the original image (height, width, channels)
                used to convert normalized coordinates back to pixel space if needed,
                but here we primarily return normalized coordinates (0.0 - 1.0) directly.

        Returns:
            Dictionary containing:
                "landmarks": A flattened 1D array of shape (128,) padded with 0s for missing hands.
                             Format: [Left_Hand(63), Right_Hand(63), Left_Presence(1), Right_Presence(1)]
                "handedness": List of strings ('Left', 'Right') for detected hands.
                "hands_detected": Integer count of detected hands.
                "raw_left": Optional[np.ndarray] shape (21, 3)
                "raw_right": Optional[np.ndarray] shape (21, 3)
        """
        raw_left: Optional[np.ndarray] = None
        raw_right: Optional[np.ndarray] = None
        handedness_list: List[str] = []

        if not self.last_results or not self.last_results.multi_hand_landmarks:
            return {
                "handedness": handedness_list,
                "hands_detected": 0,
                "raw_left": raw_left,
                "raw_right": raw_right,
            }

        for idx, hand_handedness in enumerate(self.last_results.multi_handedness):
            # MediaPipe's handedness assumes a mirrored image by default for selfie view.
            # Usually, the label is 'Left' or 'Right'.
            label = hand_handedness.classification[0].label
            handedness_list.append(label)

            hand_landmarks = self.last_results.multi_hand_landmarks[idx]
            
            # Extract 21 points * 3 coords (x, y, z)
            coords = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(hand_landmarks.landmark):
                coords[i] = [lm.x, lm.y, lm.z]

            if label == "Left":
                raw_left = coords
            else:
                raw_right = coords

        return {
            "handedness": handedness_list,
            "hands_detected": len(handedness_list),
            "raw_left": raw_left,
            "raw_right": raw_right,
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.hands:
            self.hands.close()
