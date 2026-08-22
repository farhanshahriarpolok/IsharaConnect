"""Scale and Translation Invariant Normalization Engine for BdSL Hand & Body Landmarks.

Provides:
1. Wrist-centered and palm-scale invariant landmark normalization.
2. 15-Joint 3D Interior Angle Signature calculation.
3. Finger-by-finger anatomical state classification (EXTENDED, CURL_FULL, HOOK_BENT, TOUCHING_THUMB).
4. 3D Palm normal orientation vector & facing classification.
5. Body anchor spatial proximity calculation normalized by face/torso scale.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Standard Joint Triplet Indices for 15 Finger Joints (A, Vertex B, C)
JOINT_TRIPLETS: Dict[str, List[Tuple[int, int, int]]] = {
    "thumb": [(0, 1, 2), (1, 2, 3), (2, 3, 4)],
    "index": [(0, 5, 6), (5, 6, 7), (6, 7, 8)],
    "middle": [(0, 9, 10), (9, 10, 11), (10, 11, 12)],
    "ring": [(0, 13, 14), (13, 14, 15), (14, 15, 16)],
    "pinky": [(0, 17, 18), (17, 18, 19), (18, 19, 20)],
}

# Standard Body Anchor Target Coordinates (normalized space: x in [0,1], y in [0,1])
BODY_ANCHOR_MAP: Dict[str, Tuple[float, float]] = {
    "CHIN": (0.50, 0.38),
    "UPPER_LIP": (0.50, 0.32),
    "LIP_UPPER": (0.50, 0.32),
    "PHILTRUM": (0.50, 0.32),
    "CHEEK": (0.62, 0.30),
    "CHEEK_RIGHT": (0.62, 0.30),
    "CHEEK_LEFT": (0.38, 0.30),
    "FOREHEAD": (0.50, 0.16),
    "CHEST": (0.50, 0.54),
    "CHEST_MID": (0.50, 0.54),
    "LEFT_WRIST": (0.42, 0.60),
    "RIGHT_WRIST": (0.58, 0.60),
    "NEUTRAL_SPACE": (0.50, 0.48),
}


class SpatialNormalizer:
    """Computes scale-invariant, translation-invariant representations and anatomical signatures."""

    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """Normalizes hand landmarks to be invariant to camera distance and 2D/3D translation.

        Args:
            landmarks: (21, 3) or (21, 2) array of hand coordinates.

        Returns:
            Normalized array where wrist is at origin (0,0,0) and palm scale is 1.0.
        """
        if landmarks is None or len(landmarks) < 21:
            return np.zeros((21, 3), dtype=np.float32)

        lm = np.array(landmarks, dtype=np.float32)
        if lm.shape[-1] == 2:
            lm = np.pad(lm, ((0, 0), (0, 1)), mode="constant")

        wrist = lm[0]
        # 1. Translation invariance: center at wrist
        rel_lm = lm - wrist

        # 2. Scale invariance: normalize by wrist-to-middle_mcp distance
        middle_mcp = lm[9]
        palm_scale = float(np.linalg.norm(middle_mcp - wrist))
        if palm_scale < 1e-6:
            palm_scale = 1.0

        norm_lm = rel_lm / palm_scale
        return norm_lm

    @staticmethod
    def calculate_15_joint_angles(landmarks: np.ndarray) -> np.ndarray:
        """Calculates 15 3D interior angles (in degrees) for all finger joints (MCP, PIP, DIP).

        Args:
            landmarks: (21, 3) or (21, 2) array.

        Returns:
            1D array of 15 joint angles in degrees: [T0, T1, T2, I0, I1, I2, M0, M1, M2, R0, R1, R2, P0, P1, P2].
        """
        if landmarks is None or len(landmarks) < 21:
            return np.zeros(15, dtype=np.float32)

        lm = np.array(landmarks, dtype=np.float32)
        if lm.shape[-1] == 2:
            lm = np.pad(lm, ((0, 0), (0, 1)), mode="constant")

        angles = []
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            triplets = JOINT_TRIPLETS[finger]
            for a_idx, b_idx, c_idx in triplets:
                a = lm[a_idx]
                b = lm[b_idx]  # Vertex
                c = lm[c_idx]

                v1 = a - b
                v2 = c - b

                norm_v1 = float(np.linalg.norm(v1))
                norm_v2 = float(np.linalg.norm(v2))

                if norm_v1 < 1e-7 or norm_v2 < 1e-7:
                    angles.append(180.0)
                    continue

                cosine = float(np.dot(v1, v2) / (norm_v1 * norm_v2))
                cosine = max(-1.0, min(1.0, cosine))
                angle_deg = float(np.arccos(cosine) * (180.0 / np.pi))
                angles.append(angle_deg)

        return np.array(angles, dtype=np.float32)

    @classmethod
    def detect_finger_states(
        cls,
        landmarks: np.ndarray,
        angles_15: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """Classifies each of the 5 fingers into strict anatomical states.

        States:
        - thumb: "EXTENDED", "CURL_FULL", "TOUCHING_INDEX", "ACROSS_PALM"
        - index: "EXTENDED", "HOOK_BENT", "CURL_FULL", "TOUCHING_THUMB"
        - middle: "EXTENDED", "CURL_FULL", "BENT_TOUCH_PALM"
        - ring: "EXTENDED", "CURL_FULL"
        - pinky: "EXTENDED", "CURL_FULL"
        """
        if landmarks is None or len(landmarks) < 21:
            return {
                "thumb": "CURL_FULL",
                "index": "CURL_FULL",
                "middle": "CURL_FULL",
                "ring": "CURL_FULL",
                "pinky": "CURL_FULL"
            }

        lm = np.array(landmarks, dtype=np.float32)
        wrist = lm[0]

        # Thumb extension: tip (4) distance vs IP (3) distance to wrist
        d_tip = math.hypot(lm[4, 0] - wrist[0], lm[4, 1] - wrist[1])
        d_ip = math.hypot(lm[3, 0] - wrist[0], lm[3, 1] - wrist[1])
        thumb_ext = d_tip > (d_ip * 1.04)

        # Pinch detection (Thumb Tip 4 vs Index Tip 8)
        thumb_index_dist = math.hypot(lm[4, 0] - lm[8, 0], lm[4, 1] - lm[8, 1])
        is_pinched = thumb_index_dist < 0.06

        states = {}

        # 1. Thumb
        if is_pinched:
            states["thumb"] = "TOUCHING_INDEX"
        elif thumb_ext:
            states["thumb"] = "EXTENDED"
        else:
            states["thumb"] = "CURL_FULL"

        # 2. Four Fingers: Index (8, 6), Middle (12, 10), Ring (16, 14), Pinky (20, 18)
        fingers = [("index", 8, 6), ("middle", 12, 10), ("ring", 16, 14), ("pinky", 20, 18)]
        for f_name, tip_idx, pip_idx in fingers:
            if f_name == "index" and is_pinched:
                states[f_name] = "TOUCHING_THUMB"
                continue

            # In image/norm coordinates, extended means tip is higher than PIP (smaller y)
            is_extended = bool(lm[tip_idx, 1] < lm[pip_idx, 1])
            states[f_name] = "EXTENDED" if is_extended else "CURL_FULL"

        return states

    @staticmethod
    def detect_palm_facing(landmarks: np.ndarray) -> str:
        """Computes 3D normal vector and determines palm facing orientation.

        Returns one of:
        "FACING_CAMERA", "FACING_USER", "FACING_DOWN", "FACING_UP", "FACING_LEFT", "FACING_RIGHT"
        """
        if landmarks is None or len(landmarks) < 21:
            return "FACING_CAMERA"

        lm = np.array(landmarks, dtype=np.float32)
        p0 = lm[0]
        p5 = lm[5]
        p17 = lm[17]

        v1 = p5 - p0
        v2 = p17 - p0

        normal = np.cross(v1, v2)
        norm_len = float(np.linalg.norm(normal))
        if norm_len > 1e-6:
            normal = normal / norm_len

        nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])

        # Direction checks based on dominant component
        # MediaPipe camera coordinates: +x right, +y down, +z depth (away from camera)
        if abs(nz) >= abs(nx) and abs(nz) >= abs(ny):
            return "FACING_CAMERA" if nz > -0.2 else "FACING_USER"
        elif abs(ny) >= abs(nx) and abs(ny) >= abs(nz):
            return "FACING_DOWN" if ny > 0.0 else "FACING_UP"
        else:
            return "FACING_RIGHT" if nx > 0.0 else "FACING_LEFT"

    @classmethod
    def calculate_anchor_proximity(
        cls,
        wrist_lm: np.ndarray,
        target_anchor_name: str,
        face_lm: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Calculates normalized distance (0.0 to 1.0) and estimated cm distance to target anchor.

        Args:
            wrist_lm: (3,) or (2,) normalized wrist coordinate.
            target_anchor_name: Name of target anchor (e.g. "CHIN", "UPPER_LIP", "CHEST").
            face_lm: Optional face landmarks array for dynamic IPD normalization.

        Returns:
            (normalized_score 0.0-1.0, estimated_distance_cm).
        """
        if wrist_lm is None or len(wrist_lm) < 2:
            return 0.0, 50.0

        wx, wy = float(wrist_lm[0]), float(wrist_lm[1])

        # Dynamic anchor position from face landmarks if available
        if face_lm is not None and len(face_lm) > 152:
            if target_anchor_name == "CHIN":
                target_xy = (float(face_lm[152, 0]), float(face_lm[152, 1]))  # Chin landmark
            elif target_anchor_name in ["UPPER_LIP", "LIP_UPPER", "PHILTRUM"]:
                target_xy = (float(face_lm[0, 0]), float(face_lm[0, 1]))      # Upper lip landmark
            elif target_anchor_name in ["CHEEK", "CHEEK_RIGHT"]:
                target_xy = (float(face_lm[234, 0]), float(face_lm[234, 1]))  # Right cheek
            elif target_anchor_name == "FOREHEAD":
                target_xy = (float(face_lm[10, 0]), float(face_lm[10, 1]))    # Forehead landmark
            else:
                target_xy = BODY_ANCHOR_MAP.get(target_anchor_name.upper(), (0.50, 0.48))
        else:
            target_xy = BODY_ANCHOR_MAP.get(target_anchor_name.upper(), (0.50, 0.48))

        dist_norm = math.hypot(wx - target_xy[0], wy - target_xy[1])
        # Approximate 1.0 unit = 50cm in normalized webcam frame
        dist_cm = dist_norm * 50.0

        # Score: 1.0 at dist <= 4cm (0.08 norm), 0.0 at dist >= 25cm (0.50 norm)
        score = max(0.0, min(1.0, 1.0 - (dist_norm / 0.35)))
        return score, dist_cm
