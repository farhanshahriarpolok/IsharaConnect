"""Futuristic Cyber Camera HUD Overlay for IsharaConnect.

Renders:
- Multi-colored neon glowing landmarks (Thumb: Cyan, Index: Emerald, Middle: Amber, Ring: Violet, Pinky: Rose)
- Dual-hand touch-point pulse rings when fingertips make contact
- Dynamic tracking status pill ("🟢 দুই হাত সক্রিয় | স্থিতিশীল")
- Floating sign prediction card above primary wrist
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# BGR Color Palette
COLOR_CYAN = (212, 182, 6)       # #06B6D4
COLOR_EMERALD = (129, 185, 16)   # #10B981
COLOR_AMBER = (11, 158, 245)     # #F59E0B
COLOR_VIOLET = (246, 92, 139)    # #8B5CF6
COLOR_ROSE = (94, 63, 244)       # #F43F5E
COLOR_WHITE = (248, 250, 252)    # #F8FAFC
COLOR_SLATE = (51, 65, 85)       # #334155
COLOR_DARK_BG = (15, 23, 42)     # #0F172A

FINGER_COLORS = {
    "thumb": COLOR_CYAN,
    "index": COLOR_EMERALD,
    "middle": COLOR_AMBER,
    "ring": COLOR_VIOLET,
    "pinky": COLOR_ROSE,
}

FINGER_CONNECTIONS = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring": [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky": [(0, 17), (17, 18), (18, 19), (19, 20)],
}


class CameraHUDOverlay:
    """Renders cybernetic futuristic HUD visual elements on raw video frames."""

    def __init__(self):
        self.pulse_phase = 0.0

    def draw_hud(
        self,
        frame: np.ndarray,
        left_landmarks: Optional[np.ndarray] = None,
        right_landmarks: Optional[np.ndarray] = None,
        prediction_payload: Optional[Dict[str, Any]] = None,
        fps: float = 30.0
    ) -> np.ndarray:
        """Draws glowing skeleton landmarks, HUD cards, and tracking telemetry.

        Args:
            frame: Input BGR image frame (H, W, 3).
            left_landmarks: (21, 3) normalized left hand coordinates.
            right_landmarks: (21, 3) normalized right hand coordinates.
            prediction_payload: Optional active prediction dictionary.
            fps: Current FPS reading.

        Returns:
            Annotated BGR frame.
        """
        out_frame = frame.copy()
        h, w, _ = out_frame.shape
        self.pulse_phase = (self.pulse_phase + 0.15) % (2 * math.pi)

        has_left = left_landmarks is not None and len(left_landmarks) == 21 and np.any(left_landmarks)
        has_right = right_landmarks is not None and len(right_landmarks) == 21 and np.any(right_landmarks)

        # 1. Draw Multi-Colored Glowing Landmarks for Both Hands
        if has_left:
            self._draw_hand_skeleton(out_frame, left_landmarks, w, h, is_left=True)
        if has_right:
            self._draw_hand_skeleton(out_frame, right_landmarks, w, h, is_left=False)

        # 2. Draw Dual-Hand Touch Contact Rings
        if has_left and has_right:
            self._draw_contact_rings(out_frame, left_landmarks, right_landmarks, w, h)

        # 3. Draw Top Tracking Status Pill
        self._draw_status_pill(out_frame, has_left, has_right, fps)

        # 4. Draw Floating Prediction Tag above Active Hand
        if prediction_payload:
            active_lm = right_landmarks if has_right else (left_landmarks if has_left else None)
            if active_lm is not None:
                wrist_x = int(active_lm[0][0] * w)
                wrist_y = int(active_lm[0][1] * h)
                self._draw_floating_tag(out_frame, wrist_x, wrist_y, prediction_payload)

        return out_frame

    def _draw_hand_skeleton(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        w: int,
        h: int,
        is_left: bool
    ):
        """Draws neon bones and glowing joint dots for a single hand."""
        points = []
        for lm in landmarks:
            px = int(np.clip(lm[0] * w, 0, w - 1))
            py = int(np.clip(lm[1] * h, 0, h - 1))
            points.append((px, py))

        # Palm cross connectors (0-5, 5-9, 9-13, 13-17)
        palm_lines = [(5, 9), (9, 13), (13, 17)]
        for p1_idx, p2_idx in palm_lines:
            cv2.line(frame, points[p1_idx], points[p2_idx], COLOR_SLATE, 2, cv2.LINE_AA)

        # Draw finger bones with distinct neon colors
        for finger, conns in FINGER_CONNECTIONS.items():
            color = FINGER_COLORS[finger]
            for p1_idx, p2_idx in conns:
                cv2.line(frame, points[p1_idx], points[p2_idx], color, 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx, pt in enumerate(points):
            if idx == 0:  # Wrist
                cv2.circle(frame, pt, 6, COLOR_WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 9, COLOR_CYAN, 1, cv2.LINE_AA)
            elif idx in [4, 8, 12, 16, 20]:  # Fingertips
                # Pulsing fingertip glow
                pulse_r = int(5 + 2 * math.sin(self.pulse_phase))
                cv2.circle(frame, pt, pulse_r, COLOR_WHITE, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, pulse_r + 3, COLOR_CYAN, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, COLOR_WHITE, -1, cv2.LINE_AA)

    def _draw_contact_rings(
        self,
        frame: np.ndarray,
        left_lm: np.ndarray,
        right_lm: np.ndarray,
        w: int,
        h: int
    ):
        """Draws pulsing energetic rings at fingertips when contact occurs."""
        tips = [4, 8, 12, 16, 20]
        for l_idx in tips:
            for r_idx in tips:
                p_left = np.array([left_lm[l_idx][0] * w, left_lm[l_idx][1] * h])
                p_right = np.array([right_lm[r_idx][0] * w, right_lm[r_idx][1] * h])
                dist = np.linalg.norm(p_left - p_right)
                if dist < 35.0:  # Contact active
                    mid = ((p_left + p_right) / 2.0).astype(int)
                    ring_r = int(12 + 6 * math.sin(self.pulse_phase))
                    cv2.circle(frame, (mid[0], mid[1]), ring_r, COLOR_EMERALD, 2, cv2.LINE_AA)
                    cv2.circle(frame, (mid[0], mid[1]), ring_r + 5, COLOR_CYAN, 1, cv2.LINE_AA)

    def _draw_status_pill(
        self,
        frame: np.ndarray,
        has_left: bool,
        has_right: bool,
        fps: float
    ):
        """Draws top tracking telemetry badge."""
        if has_left and has_right:
            status_text = "DUAL HAND ACTIVE"
            pill_color = COLOR_EMERALD
        elif has_left or has_right:
            status_text = "SINGLE HAND ACTIVE"
            pill_color = COLOR_CYAN
        else:
            status_text = "SEARCHING FOR HANDS..."
            pill_color = COLOR_ROSE

        # Draw dark translucent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (320, 44), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Draw glowing accent indicator
        cv2.circle(frame, (26, 28), 5, pill_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{status_text} | {fps:.1f} FPS", (40, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_floating_tag(
        self,
        frame: np.ndarray,
        wrist_x: int,
        wrist_y: int,
        payload: Dict[str, Any]
    ):
        """Draws floating cybernetic prediction label above the wrist."""
        label_bn = payload.get("label_bn", "")
        confidence = payload.get("confidence", 0.0)
        source = payload.get("source", "engine")

        card_x = max(10, min(wrist_x - 70, frame.shape[1] - 180))
        card_y = max(55, wrist_y - 30)

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y - 28), (card_x + 160, card_y + 8), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border
        border_color = COLOR_EMERALD if confidence >= 0.80 else COLOR_CYAN
        cv2.rectangle(frame, (card_x, card_y - 28), (card_x + 160, card_y + 8), border_color, 1, cv2.LINE_AA)

        # Text
        display_str = f"{label_bn} ({int(confidence * 100)}%)"
        cv2.putText(frame, display_str, (card_x + 8, card_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
