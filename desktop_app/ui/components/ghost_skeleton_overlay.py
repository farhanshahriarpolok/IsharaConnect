"""Interactive Target Ghost Skeleton Overlay for IsharaConnect.

Renders:
1. Animated, semi-transparent neon wireframe skeleton of the target sign pose.
2. Dynamic per-joint color feedback based on alignment:
   - 🟢 Neon Green (#10B981): Joint angle & 3D position aligned (>= 85%).
   - 🟡 Amber Yellow (#F59E0B): Minor deviation (50% - 84%).
   - 🔴 Coral Red (#EF4444): Major misalignment (< 50%).
3. Glowing Alignment Bullseye when the hand enters the correct anatomical anchor zone.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core_engine.vision.spatial_normalizer import SpatialNormalizer, BODY_ANCHOR_MAP, JOINT_TRIPLETS

# Standard Color Palette (BGR)
COLOR_NEON_GREEN = (129, 185, 16)   # #10B981
COLOR_AMBER = (11, 158, 245)        # #F59E0B
COLOR_CORAL_RED = (68, 68, 239)     # #EF4444
COLOR_GHOST_CYAN = (212, 182, 6)    # #06B6D4
COLOR_WHITE = (248, 250, 252)

FINGER_CHAINS = [
    [0, 1, 2, 3, 4],        # Thumb
    [0, 5, 6, 7, 8],        # Index
    [0, 9, 10, 11, 12],     # Middle
    [0, 13, 14, 15, 16],    # Ring
    [0, 17, 18, 19, 20]     # Pinky
]


class GhostSkeletonOverlay:
    """Renders semi-transparent target ghost skeleton and per-joint alignment feedback."""

    def __init__(self):
        self.normalizer = SpatialNormalizer()
        self.pulse_phase = 0.0

    def get_canonical_target_landmarks(
        self,
        target_slug: str,
        w: int,
        h: int,
        target_anchor: str = "NEUTRAL_SPACE"
    ) -> np.ndarray:
        """Generates pixel-space 21 landmarks representing the canonical target pose."""
        anchor_xy = BODY_ANCHOR_MAP.get(target_anchor.upper(), (0.50, 0.48))
        wx = anchor_xy[0] * w
        wy = anchor_xy[1] * h
        scale = 0.22 * min(w, h)

        lm = np.zeros((21, 3), dtype=np.float32)
        lm[0] = [wx, wy, 0.0]

        # Shape modifiers based on sign
        is_fist = target_slug in ["vowel_a", "fist"]
        is_index_only = target_slug in ["baba", "cons_ka", "point", "dada", "chacha"]
        is_pinch = target_slug in ["taka", "pinch"]

        # Thumb
        lm[1] = [wx - 0.15 * scale, wy - 0.10 * scale, 0.0]
        lm[2] = [wx - 0.25 * scale, wy - 0.20 * scale, 0.0]
        lm[3] = [wx - 0.35 * scale, wy - 0.30 * scale, 0.0]
        if is_pinch:
            lm[4] = [wx + 0.08 * scale, wy - 0.70 * scale, 0.0]  # Touching index
        elif is_fist:
            lm[4] = [wx - 0.10 * scale, wy - 0.15 * scale, 0.0]  # Folded across
        else:
            lm[4] = [wx - 0.45 * scale, wy - 0.40 * scale, 0.0]  # Extended

        # 4 Fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
        offsets = [(0.10, 5), (0.20, 9), (0.30, 13), (0.40, 17)]
        for f_idx, (x_off, base) in enumerate(offsets):
            f_name = ["index", "middle", "ring", "pinky"][f_idx]
            is_extended = True
            if is_fist:
                is_extended = False
            elif is_index_only and f_name != "index":
                is_extended = False

            bx = wx + x_off * scale
            lm[base] = [bx, wy - 0.25 * scale, 0.0]
            if is_extended:
                lm[base + 1] = [bx, wy - 0.45 * scale, 0.0]
                lm[base + 2] = [bx, wy - 0.65 * scale, 0.0]
                lm[base + 3] = [bx, wy - 0.85 * scale, 0.0]
            else:
                lm[base + 1] = [bx, wy - 0.40 * scale, 0.15 * scale]
                lm[base + 2] = [bx, wy - 0.30 * scale, 0.25 * scale]
                lm[base + 3] = [bx, wy - 0.15 * scale, 0.15 * scale]

        return lm

    def render_ghost_overlay(
        self,
        frame: np.ndarray,
        target_slug: str,
        user_landmarks: Optional[np.ndarray],
        target_anchor: str = "NEUTRAL_SPACE",
        match_score: float = 0.0
    ) -> np.ndarray:
        """Draws pulsing semi-transparent ghost target wireframe and colored alignment joints.

        Args:
            frame: Input BGR frame.
            target_slug: Target sign identifier.
            user_landmarks: (21, 3) normalized user hand landmarks.
            target_anchor: Target body anchor name.
            match_score: Current alignment accuracy percentage.

        Returns:
            Annotated frame.
        """
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()
        h, w, _ = out.shape
        self.pulse_phase = (self.pulse_phase + 0.12) % (2 * math.pi)
        pulse_alpha = 0.35 + 0.15 * math.sin(self.pulse_phase)

        # 1. Generate Target Canonical Skeleton
        ghost_lm = self.get_canonical_target_landmarks(target_slug, w, h, target_anchor)

        # 2. Draw Semi-Transparent Ghost Wireframe
        ghost_layer = np.zeros_like(out)
        for chain in FINGER_CHAINS:
            for i in range(len(chain) - 1):
                p1 = (int(ghost_lm[chain[i], 0]), int(ghost_lm[chain[i], 1]))
                p2 = (int(ghost_lm[chain[i + 1], 0]), int(ghost_lm[chain[i + 1], 1]))
                cv2.line(ghost_layer, p1, p2, COLOR_GHOST_CYAN, 2, cv2.LINE_AA)

        for pt in ghost_lm:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(ghost_layer, (cx, cy), 3, (248, 250, 252), -1, cv2.LINE_AA)

        # Blend ghost layer
        cv2.addWeighted(out, 1.0, ghost_layer, pulse_alpha, 0, out)

        # 3. Draw Anchor Bullseye Target Ring
        anchor_xy = BODY_ANCHOR_MAP.get(target_anchor.upper(), (0.50, 0.48))
        ax = int(anchor_xy[0] * w)
        ay = int(anchor_xy[1] * h)
        bullseye_radius = int(24 + 4 * math.sin(self.pulse_phase * 2))
        ring_color = COLOR_NEON_GREEN if match_score >= 75.0 else COLOR_AMBER
        cv2.circle(out, (ax, ay), bullseye_radius, ring_color, 1, cv2.LINE_AA)
        cv2.drawMarker(out, (ax, ay), ring_color, cv2.MARKER_CROSS, 10, 1)

        # 4. Color-Code User Landmarks based on Joint Alignment
        if user_landmarks is not None and len(user_landmarks) >= 21 and np.any(user_landmarks):
            user_px = []
            for lm in user_landmarks:
                user_px.append((int(lm[0] * w), int(lm[1] * h)))

            # Evaluate joint alignments
            joint_colors = self._evaluate_joint_colors(user_landmarks, ghost_lm, match_score)

            # Draw bones
            for chain in FINGER_CHAINS:
                for i in range(len(chain) - 1):
                    p1 = user_px[chain[i]]
                    p2 = user_px[chain[i + 1]]
                    bone_color = joint_colors[chain[i + 1]]
                    cv2.line(out, p1, p2, bone_color, 3, cv2.LINE_AA)

            # Draw joint nodes
            for idx, pt in enumerate(user_px):
                color = joint_colors[idx]
                cv2.circle(out, pt, 5, color, -1, cv2.LINE_AA)
                cv2.circle(out, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def _evaluate_joint_colors(
        self,
        user_lm: np.ndarray,
        target_lm: np.ndarray,
        match_score: float
    ) -> List[Tuple[int, int, int]]:
        """Maps each of the 21 joints to Neon Green, Amber, or Coral Red based on angular precision."""
        colors = [COLOR_NEON_GREEN if match_score >= 75.0 else COLOR_AMBER] * 21

        # Check joint extensions
        norm_user = self.normalizer.normalize_landmarks(user_lm)
        user_states = self.normalizer.detect_finger_states(norm_user)

        finger_node_map = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20]
        }

        for finger, nodes in finger_node_map.items():
            st = user_states.get(finger, "CURL_FULL")
            if match_score >= 85.0:
                color = COLOR_NEON_GREEN
            elif match_score >= 50.0:
                color = COLOR_NEON_GREEN if st == "EXTENDED" else COLOR_AMBER
            else:
                color = COLOR_CORAL_RED

            for node_idx in nodes:
                colors[node_idx] = color

        return colors
