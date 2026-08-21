"""Dual-Hand Spatial Feature Engine.

Extracts 42 3D landmarks for both hands, computes dual-wrist midpoint normalization,
inter-hand finger distances, and basic orientation vectors.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)

class SpatialHandEngine:
    def __init__(self, static_image_mode: bool = False):
        try:
            from mediapipe.python.solutions import hands as mp_hands
        except (ImportError, AttributeError):
            import mediapipe.solutions.hands as mp_hands
            
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.fingertips = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        
    def extract_spatial_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract spatial features from an RGB frame.
        
        Returns:
            dict containing:
                - 'landmarks': 42x3 array of (x,y,z) coordinates. Left hand first (0-20), Right hand (21-41). Zeros if missing.
                - 'normalized_landmarks': Landmarks normalized relative to the dual-wrist midpoint (if both hands visible) or single wrist.
                - 'touch_matrix': 5x5 euclidean distance matrix between left fingertips and right fingertips.
                - 'orientation': {'left': [yaw, pitch, roll], 'right': [yaw, pitch, roll]}
        """
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Initialize empty arrays
        raw_landmarks = np.zeros((42, 3), dtype=np.float32)
        
        has_left = False
        has_right = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # MediaPipe labels are mirrored by default for front-facing camera unless flipped, 
                # but let's trust the label 'Left' or 'Right'
                label = handedness.classification[0].label
                idx_offset = 0 if label == 'Left' else 21
                
                if label == 'Left':
                    has_left = True
                else:
                    has_right = True
                    
                for i, lm in enumerate(hand_landmarks.landmark):
                    raw_landmarks[idx_offset + i] = [lm.x, lm.y, lm.z]
                    
        # 1. Coordinate normalization
        normalized_landmarks = np.copy(raw_landmarks)
        if has_left and has_right:
            midpoint = (raw_landmarks[0] + raw_landmarks[21]) / 2.0
            normalized_landmarks[0:21] -= midpoint
            normalized_landmarks[21:42] -= midpoint
        elif has_left:
            midpoint = raw_landmarks[0]
            normalized_landmarks[0:21] -= midpoint
        elif has_right:
            midpoint = raw_landmarks[21]
            normalized_landmarks[21:42] -= midpoint
            
        # 2. Inter-hand finger-touch distance matrix (5x5)
        touch_matrix = np.full((5, 5), float('inf'), dtype=np.float32)
        if has_left and has_right:
            for i, left_idx in enumerate(self.fingertips):
                for j, right_idx in enumerate(self.fingertips):
                    p1 = raw_landmarks[left_idx]
                    p2 = raw_landmarks[21 + right_idx]
                    dist = np.linalg.norm(p1 - p2)
                    touch_matrix[i, j] = float(dist)
                    
        # 3. Hand orientation vectors (proxy)
        # We can calculate a crude orientation by looking at Wrist(0) -> MCP(9) 
        # and Wrist(0) -> IndexMCP(5) vs PinkyMCP(17) cross product
        orientation = {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]}
        
        def calculate_orientation(offset):
            wrist = raw_landmarks[offset + 0]
            mcp9 = raw_landmarks[offset + 9]
            mcp5 = raw_landmarks[offset + 5]
            mcp17 = raw_landmarks[offset + 17]
            
            # Forward vector
            forward = mcp9 - wrist
            forward_norm = np.linalg.norm(forward)
            if forward_norm > 1e-6:
                forward = forward / forward_norm
                
            # Right vector (across palm)
            right = mcp17 - mcp5
            right_norm = np.linalg.norm(right)
            if right_norm > 1e-6:
                right = right / right_norm
                
            # Up vector (palm normal)
            up = np.cross(right, forward)
            up_norm = np.linalg.norm(up)
            if up_norm > 1e-6:
                up = up / up_norm
                
            # Convert to simple pitch/yaw/roll proxies based on vector components
            pitch = np.arcsin(np.clip(forward[1], -1.0, 1.0))
            yaw = np.arctan2(forward[0], forward[2])
            roll = np.arctan2(right[1], right[0])
            return [float(yaw), float(pitch), float(roll)]
            
        if has_left:
            orientation["left"] = calculate_orientation(0)
        if has_right:
            orientation["right"] = calculate_orientation(21)

        # 4. 151D flattened feature vector (126 coordinates + 25 touch distances)
        touch_clean = np.where(np.isinf(touch_matrix), 1.0, touch_matrix).flatten()
        spatial_vector = np.concatenate([normalized_landmarks.flatten(), touch_clean]).astype(np.float32)
            
        return {
            "has_left": has_left,
            "has_right": has_right,
            "raw_landmarks": raw_landmarks,
            "normalized_landmarks": normalized_landmarks,
            "touch_matrix": touch_matrix,
            "orientation": orientation,
            "spatial_vector": spatial_vector
        }
