"""Ghost Overlay Component for interactive spatial guidance."""

import logging
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QPixmap

logger = logging.getLogger(__name__)

# MediaPipe Hand Connections (simplified for drawing)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]

class GhostOverlayPainter:
    """Renders translucent reference skeletons and calculates live overlap."""
    
    def __init__(self):
        # Base colors
        self.ghost_color = QColor(137, 180, 250, 150) # Translucent Cyan (#89B4FA)
        self.match_color = QColor(16, 185, 129, 200)  # Bright Emerald (#10B981)
        self.tolerance_radius_ratio = 0.10            # 10% tolerance
        
    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def calculate_alignment_score(self, ref_landmarks: List[Tuple[float, float]], live_landmarks: List[Tuple[float, float]]) -> float:
        """Calculate spatial overlap percentage between live and reference landmarks."""
        if not ref_landmarks or not live_landmarks or len(ref_landmarks) != len(live_landmarks):
            return 0.0
            
        matches = 0
        total = len(ref_landmarks)
        
        # Determine tolerance based on reference bounding box size
        xs = [p[0] for p in ref_landmarks]
        ys = [p[1] for p in ref_landmarks]
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)
        tolerance = max(box_width, box_height) * self.tolerance_radius_ratio
        
        # Minimum absolute tolerance in pixels just in case box is tiny
        tolerance = max(tolerance, 15.0)
        
        for ref_pt, live_pt in zip(ref_landmarks, live_landmarks):
            dist = self._calculate_distance(ref_pt, live_pt)
            if dist <= tolerance:
                matches += 1
                
        return (matches / total) * 100.0

    def draw_overlay(self, image: QImage, ref_landmarks: List[Tuple[float, float]], live_landmarks: Optional[List[Tuple[float, float]]] = None) -> QImage:
        """Draws the ghost skeleton onto the camera frame.
        
        Args:
            image: Original camera frame (QImage)
            ref_landmarks: Expected landmarks (absolute pixel coords)
            live_landmarks: User's live landmarks (absolute pixel coords)
            
        Returns:
            QImage with overlay rendered
        """
        if not ref_landmarks:
            return image
            
        # Draw on a copy
        result_img = image.copy()
        painter = QPainter(result_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        score = self.calculate_alignment_score(ref_landmarks, live_landmarks) if live_landmarks else 0.0
        
        # If score > 80%, we treat the pose as matching and color the ghost green
        current_color = self.match_color if score > 80.0 else self.ghost_color
        
        pen_line = QPen(current_color, 4, Qt.PenStyle.SolidLine)
        pen_joint = QPen(current_color, 6, Qt.PenStyle.SolidLine)
        
        # Draw connections
        painter.setPen(pen_line)
        num_landmarks = len(ref_landmarks)
        
        # Handle left and right hands if both are present in the flat list
        for offset in [0, 21]:
            if offset + 20 >= num_landmarks:
                break
                
            for conn in HAND_CONNECTIONS:
                idx1 = offset + conn[0]
                idx2 = offset + conn[1]
                
                if idx1 < num_landmarks and idx2 < num_landmarks:
                    p1 = ref_landmarks[idx1]
                    p2 = ref_landmarks[idx2]
                    painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))
                    
        # Draw joints
        painter.setPen(pen_joint)
        for i, pt in enumerate(ref_landmarks):
            painter.drawPoint(QPointF(pt[0], pt[1]))
            
        painter.end()
        return result_img
