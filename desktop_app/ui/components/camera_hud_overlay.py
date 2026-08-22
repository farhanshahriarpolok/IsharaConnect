"""Futuristic Cyber Camera HUD Overlay for IsharaConnect.

Renders:
- Multi-colored neon glowing landmarks (Thumb: Cyan, Index: Emerald, Middle: Amber, Ring: Violet, Pinky: Rose)
- Dual-hand touch-point pulse rings when fingertips make contact
- Dynamic tracking status pill ("🟢 দুই হাত সক্রিয় | স্থিতিশীল")
- Floating sign prediction card above primary wrist
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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


def _safe_conf(val: Optional[Union[float, int]]) -> float:
    """Universal NaN, Inf, and range sanitization helper for confidence scores."""
    if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
        return 0.0
    return max(0.0, min(1.0, float(val)))


from desktop_app.ui.components.ghost_skeleton_overlay import GhostSkeletonOverlay


class CameraHUDOverlay:
    """Renders cybernetic futuristic HUD visual elements on raw video frames."""

    def __init__(self):
        self.pulse_phase = 0.0
        self.ghost_overlay = GhostSkeletonOverlay()

    def draw_hud(
        self,
        frame: np.ndarray,
        left_landmarks: Optional[np.ndarray] = None,
        right_landmarks: Optional[np.ndarray] = None,
        prediction_payload: Optional[Dict[str, Any]] = None,
        fps: float = 30.0,
        diagnostic_result: Optional[Any] = None,
        ghost_target_slug: Optional[str] = None,
        target_anchor: str = "NEUTRAL_SPACE"
    ) -> np.ndarray:
        """Draws glowing skeleton landmarks, HUD cards, and tracking telemetry.

        Args:
            frame: Input BGR image frame (H, W, 3).
            left_landmarks: (21, 3) normalized left hand coordinates.
            right_landmarks: (21, 3) normalized right hand coordinates.
            prediction_payload: Optional active prediction dictionary.
            fps: Current FPS reading.
            diagnostic_result: Optional DiagnosticResult instance from SignCorrectionAdvisor.
            ghost_target_slug: Optional target sign slug for rendering reference ghost skeleton.
            target_anchor: Optional target anchor name.

        Returns:
            Annotated BGR frame.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 3:
            return frame

        out_frame = frame.copy()
        h, w, _ = out_frame.shape
        self.pulse_phase = (self.pulse_phase + 0.15) % (2 * math.pi)

        # Sanitize FPS
        safe_fps = 0.0 if (fps is None or not isinstance(fps, (int, float)) or math.isnan(fps) or math.isinf(fps) or fps < 0) else float(fps)

        has_left = (
            left_landmarks is not None
            and len(left_landmarks) == 21
            and not np.isnan(left_landmarks).any()
            and np.any(left_landmarks)
        )
        has_right = (
            right_landmarks is not None
            and len(right_landmarks) == 21
            and not np.isnan(right_landmarks).any()
            and np.any(right_landmarks)
        )

        # 1. Render Interactive Target Ghost Skeleton if enabled
        active_target = ghost_target_slug or (getattr(diagnostic_result, "target_gloss", None) if diagnostic_result else None)
        match_score = getattr(diagnostic_result, "match_score", 0.0) if diagnostic_result else 0.0
        active_user_lm = right_landmarks if has_right else (left_landmarks if has_left else None)

        if active_target:
            out_frame = self.ghost_overlay.render_ghost_overlay(
                out_frame,
                active_target,
                active_user_lm,
                target_anchor,
                match_score
            )
        else:
            # Standard landmarks fallback
            if has_left:
                self._draw_hand_skeleton(out_frame, left_landmarks, w, h, is_left=True)
            if has_right:
                self._draw_hand_skeleton(out_frame, right_landmarks, w, h, is_left=False)

        # 2. Draw Dual-Hand Touch Contact Rings
        if has_left and has_right:
            self._draw_contact_rings(out_frame, left_landmarks, right_landmarks, w, h)

        # 3. Draw Top Tracking Status Pill
        self._draw_status_pill(out_frame, has_left, has_right, safe_fps)

        # 4. Draw 4-Channel Articulatory Diagnostic Status Chips & Advice
        if diagnostic_result is not None:
            self._draw_diagnostic_chips(out_frame, diagnostic_result, w, h)

        # 5. Draw Floating Prediction Tag above Active Hand
        if prediction_payload and isinstance(prediction_payload, dict):
            active_lm = right_landmarks if has_right else (left_landmarks if has_left else None)
            if active_lm is not None and len(active_lm) > 0:
                raw_wx = float(active_lm[0][0]) * w
                raw_wy = float(active_lm[0][1]) * h
                if not (math.isnan(raw_wx) or math.isinf(raw_wx) or math.isnan(raw_wy) or math.isinf(raw_wy)):
                    wrist_x = int(np.clip(raw_wx, 0, w - 1))
                    wrist_y = int(np.clip(raw_wy, 0, h - 1))
                    self._draw_floating_tag(out_frame, wrist_x, wrist_y, prediction_payload)
            else:
                # If no hands detected, draw floating tag at top-left fallback position
                self._draw_floating_tag(out_frame, 80, 80, prediction_payload)

        return out_frame

    def _draw_diagnostic_chips(
        self,
        frame: np.ndarray,
        diag: Any,
        w: int,
        h: int
    ):
        """Draws 4-channel diagnostic telemetry chips and 4-row status card."""
        channel_status = getattr(diag, "channel_status", {}) or {}
        hints = getattr(diag, "corrective_hints", []) or []
        match_score = getattr(diag, "match_score", 0.0)
        checklist_rows = getattr(diag, "checklist_rows", []) or []

        # 1. Top-Left 4-Row Diagnostic HUD Status Card
        if checklist_rows:
            card_x = 12
            card_y = 52
            card_w = 175
            card_h = len(checklist_rows) * 20 + 26

            # Draw background overlay card
            sub_roi = frame[max(0, card_y): min(h, card_y + card_h), max(0, card_x): min(w, card_x + card_w)]
            if sub_roi.size > 0:
                dark_card = np.full_like(sub_roi, (15, 23, 42), dtype=np.uint8)
                cv2.addWeighted(sub_roi, 0.20, dark_card, 0.80, 0, sub_roi)
                cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (51, 65, 85), 1)

                # Card Header
                cv2.putText(
                    frame,
                    f"DIAGNOSTIC COACH ({int(match_score)}%)",
                    (card_x + 8, card_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (6, 182, 212),
                    1,
                    cv2.LINE_AA
                )

                for idx, row in enumerate(checklist_rows):
                    ry = card_y + 32 + idx * 19
                    st = row.get("status", "ok")
                    color = (16, 185, 129) if st == "ok" else ((11, 158, 245) if st == "warn" else (94, 63, 244))
                    icon = "OK" if st == "ok" else ("!" if st == "warn" else "X")
                    title = row.get("title", "")
                    text = f"{row.get('icon', '')} {title}: {icon}"
                    cv2.putText(
                        frame,
                        text,
                        (card_x + 8, ry),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        color,
                        1,
                        cv2.LINE_AA
                    )

        # 2. Bottom 4-Channel Horizontal Chip Bar
        chips = [
            ("Hand", "handedness"),
            ("Pos", "position"),
            ("Fingers", "fingers"),
            ("Palm", "orientation")
        ]

        chip_w = 70
        chip_h = 22
        start_x = 12
        y = h - chip_h - 12

        # Draw semi-transparent background bar
        bar_bg = frame[max(0, y - 4): min(h, y + chip_h + 4), 8: min(w, 8 + len(chips) * (chip_w + 6) + 120)]
        if bar_bg.size > 0:
            dark_overlay = np.full_like(bar_bg, (15, 23, 42), dtype=np.uint8)
            cv2.addWeighted(bar_bg, 0.25, dark_overlay, 0.75, 0, bar_bg)

        for idx, (label, key) in enumerate(chips):
            cx = start_x + idx * (chip_w + 6)
            status = channel_status.get(key, "ok")
            if status == "ok":
                color_bg = (16, 185, 129)  # Green
                icon = "OK"
            elif status == "warn":
                color_bg = (11, 158, 245)  # Amber
                icon = "!"
            else:
                color_bg = (94, 63, 244)   # Rose
                icon = "X"

            # Draw chip box
            cv2.rectangle(frame, (cx, y), (cx + chip_w, y + chip_h), (15, 23, 42), -1)
            cv2.rectangle(frame, (cx, y), (cx + chip_w, y + chip_h), color_bg, 1)

            text_str = f"{label}: {icon}"
            cv2.putText(
                frame,
                text_str,
                (cx + 4, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (248, 250, 252),
                1,
                cv2.LINE_AA
            )

        # Draw Match Accuracy Score Chip
        score_x = start_x + len(chips) * (chip_w + 6) + 4
        score_color = (16, 185, 129) if match_score >= 75.0 else ((11, 158, 245) if match_score >= 50.0 else (94, 63, 244))
        cv2.rectangle(frame, (score_x, y), (score_x + 90, y + chip_h), (15, 23, 42), -1)
        cv2.rectangle(frame, (score_x, y), (score_x + 90, y + chip_h), score_color, 1)
        cv2.putText(
            frame,
            f"Match: {int(match_score)}%",
            (score_x + 6, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            score_color,
            1,
            cv2.LINE_AA
        )

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
            raw_px = float(lm[0]) * w
            raw_py = float(lm[1]) * h
            if math.isnan(raw_px) or math.isinf(raw_px):
                raw_px = 0.0
            if math.isnan(raw_py) or math.isinf(raw_py):
                raw_py = 0.0
            px = int(np.clip(raw_px, 0, w - 1))
            py = int(np.clip(raw_py, 0, h - 1))
            points.append((px, py))

        if len(points) < 21:
            return

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
                if l_idx >= len(left_lm) or r_idx >= len(right_lm):
                    continue
                p_left = np.array([left_lm[l_idx][0] * w, left_lm[l_idx][1] * h])
                p_right = np.array([right_lm[r_idx][0] * w, right_lm[r_idx][1] * h])
                if np.any(np.isnan(p_left)) or np.any(np.isinf(p_left)) or np.any(np.isnan(p_right)) or np.any(np.isinf(p_right)):
                    continue
                dist = np.linalg.norm(p_left - p_right)
                if dist < 35.0:  # Contact active
                    mid = ((p_left + p_right) / 2.0).astype(int)
                    ring_r = int(12 + 6 * math.sin(self.pulse_phase))
                    cv2.circle(frame, (int(mid[0]), int(mid[1])), ring_r, COLOR_EMERALD, 2, cv2.LINE_AA)
                    cv2.circle(frame, (int(mid[0]), int(mid[1])), ring_r + 5, COLOR_CYAN, 1, cv2.LINE_AA)

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
        raw_conf = payload.get("confidence", 0.0)
        confidence = _safe_conf(raw_conf)
        source = payload.get("source", "engine")

        card_x = max(10, min(wrist_x - 70, frame.shape[1] - 180))
        card_y = max(55, min(wrist_y - 30, frame.shape[0] - 10))

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y - 28), (card_x + 160, card_y + 8), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border
        border_color = COLOR_EMERALD if confidence >= 0.80 else COLOR_CYAN
        cv2.rectangle(frame, (card_x, card_y - 28), (card_x + 160, card_y + 8), border_color, 1, cv2.LINE_AA)

        # Text with universal NaN guard
        display_str = f"{label_bn} ({int(confidence * 100)}%)"
        cv2.putText(frame, display_str, (card_x + 8, card_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
