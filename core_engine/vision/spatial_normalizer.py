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
    def calculate_finger_angles(cls, landmarks: np.ndarray) -> np.ndarray:
        """Alias for calculate_15_joint_angles returning continuous 3D interior degrees."""
        return cls.calculate_15_joint_angles(landmarks)

    @classmethod
    def get_finger_angles_dict(cls, landmarks: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Returns structured dictionary of continuous degree angles for all 5 fingers.

        Format:
        {
          "thumb": {"mcp": deg, "ip": deg, "tip": deg},
          "index": {"mcp": deg, "pip": deg, "dip": deg},
          "middle": {"mcp": deg, "pip": deg, "dip": deg},
          "ring": {"mcp": deg, "pip": deg, "dip": deg},
          "pinky": {"mcp": deg, "pip": deg, "dip": deg}
        }
        """
        angles_15 = cls.calculate_15_joint_angles(landmarks)
        return {
            "thumb": {"mcp": float(angles_15[0]), "ip": float(angles_15[1]), "tip": float(angles_15[2])},
            "index": {"mcp": float(angles_15[3]), "pip": float(angles_15[4]), "dip": float(angles_15[5])},
            "middle": {"mcp": float(angles_15[6]), "pip": float(angles_15[7]), "dip": float(angles_15[8])},
            "ring": {"mcp": float(angles_15[9]), "pip": float(angles_15[10]), "dip": float(angles_15[11])},
            "pinky": {"mcp": float(angles_15[12]), "pip": float(angles_15[13]), "dip": float(angles_15[14])}
        }

    @staticmethod
    def smooth_landmarks(
        raw_landmarks: Optional[np.ndarray],
        history: Optional[np.ndarray],
        alpha: float = 0.65
    ) -> np.ndarray:
        """Applies Exponential Moving Average (EMA) smoothing to eliminate camera sensor jitter.

        Args:
            raw_landmarks: Current frame raw (21, 3) landmarks.
            history: Previous frame smoothed (21, 3) landmarks.
            alpha: Smoothing weight factor [0.0 - 1.0]. Higher = more responsive, Lower = smoother.

        Returns:
            Smoothed (21, 3) landmarks array.
        """
        if raw_landmarks is None or len(raw_landmarks) == 0:
            return history if history is not None else np.zeros((21, 3), dtype=np.float32)

        raw = np.array(raw_landmarks, dtype=np.float32)
        if history is None or history.shape != raw.shape or np.isnan(history).any():
            return raw

        # EMA formula: L_t = alpha * raw + (1 - alpha) * history
        smoothed = alpha * raw + (1.0 - alpha) * history
        return smoothed.astype(np.float32)

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
    def get_anatomical_anchor_3d(
        cls,
        anchor_name: str,
        face_landmarks: Optional[np.ndarray] = None,
        pose_landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Resolves the exact 3D biometric landmark coordinate for a target body anchor.

        MediaPipe FaceMesh Keypoint Clusters:
        - FOREHEAD / GLABELLA: Indices 10, 151, 9 (Mid Forehead)
        - NOSE: Indices 1, 4, 195 (Nose Tip / Bridge)
        - UPPER_LIP / PHILTRUM: Indices 0, 13, 267
        - CHIN / LOWER_CHIN: Indices 152, 175, 199 (Mental protuberance)
        - CHEEK_RIGHT: Indices 234, 93 | CHEEK_LEFT: Indices 454, 323
        - CHEST_MID: Pose 11 & 12 midpoint (Mid Sternum)
        - LEFT_WRIST: Pose 15
        """
        name = anchor_name.upper()

        # 1. FaceMesh 3D Biometric Keypoint Clusters
        if face_landmarks is not None and len(face_landmarks) >= 153:
            flm = np.array(face_landmarks, dtype=np.float32)
            if flm.shape[-1] == 2:
                flm = np.pad(flm, ((0, 0), (0, 1)), mode="constant")

            if name in ["CHIN", "LOWER_CHIN"]:
                cluster = [flm[152]]
                if len(flm) > 175 and np.any(flm[175]):
                    cluster.append(flm[175])
                if len(flm) > 199 and np.any(flm[199]):
                    cluster.append(flm[199])
                return np.mean(cluster, axis=0)
            elif name in ["UPPER_LIP", "LIP_UPPER", "PHILTRUM"]:
                cluster = [flm[0]]
                if len(flm) > 13 and np.any(flm[13]):
                    cluster.append(flm[13])
                if len(flm) > 267 and np.any(flm[267]):
                    cluster.append(flm[267])
                return np.mean(cluster, axis=0)
            elif name in ["NOSE", "NOSE_TIP"]:
                cluster = [flm[1]]
                if len(flm) > 4 and np.any(flm[4]):
                    cluster.append(flm[4])
                if len(flm) > 195 and np.any(flm[195]):
                    cluster.append(flm[195])
                return np.mean(cluster, axis=0)
            elif name in ["FOREHEAD", "GLABELLA"]:
                cluster = [flm[10]]
                if len(flm) > 151 and np.any(flm[151]):
                    cluster.append(flm[151])
                if len(flm) > 9 and np.any(flm[9]):
                    cluster.append(flm[9])
                return np.mean(cluster, axis=0)
            elif name in ["CHEEK", "CHEEK_RIGHT"]:
                cluster = [flm[234]] if (len(flm) > 234 and np.any(flm[234])) else [np.array([0.62, 0.30, 0.0], dtype=np.float32)]
                if len(flm) > 93 and np.any(flm[93]):
                    cluster.append(flm[93])
                return np.mean(cluster, axis=0)
            elif name == "CHEEK_LEFT":
                cluster = [flm[454]] if (len(flm) > 454 and np.any(flm[454])) else [np.array([0.38, 0.30, 0.0], dtype=np.float32)]
                if len(flm) > 323 and np.any(flm[323]):
                    cluster.append(flm[323])
                return np.mean(cluster, axis=0)

        # 2. Pose 3D Biometric Landmarks
        if pose_landmarks is not None and len(pose_landmarks) >= 16:
            plm = np.array(pose_landmarks, dtype=np.float32)
            if plm.shape[-1] == 2:
                plm = np.pad(plm, ((0, 0), (0, 1)), mode="constant")

            if name in ["CHEST", "CHEST_MID"]:
                return (plm[11] + plm[12]) / 2.0  # Mid sternum
            elif name == "LEFT_WRIST":
                return plm[15]
            elif name == "RIGHT_WRIST":
                return plm[16]

        # 3. Standard Fallbacks
        fallback_map = {
            "CHIN": (0.50, 0.44),
            "LOWER_CHIN": (0.50, 0.44),
            "UPPER_LIP": (0.50, 0.38),
            "LIP_UPPER": (0.50, 0.38),
            "PHILTRUM": (0.50, 0.38),
            "NOSE": (0.50, 0.32),
            "FOREHEAD": (0.50, 0.20),
            "GLABELLA": (0.50, 0.20),
            "CHEEK": (0.62, 0.34),
            "CHEEK_RIGHT": (0.62, 0.34),
            "CHEEK_LEFT": (0.38, 0.34),
            "CHEST": (0.50, 0.60),
            "CHEST_MID": (0.50, 0.60),
            "LEFT_WRIST": (0.42, 0.65),
            "RIGHT_WRIST": (0.58, 0.65),
            "NEUTRAL_SPACE": (0.50, 0.50)
        }
        xy = fallback_map.get(name, BODY_ANCHOR_MAP.get(name, (0.50, 0.48)))
        return np.array([xy[0], xy[1], 0.0], dtype=np.float32)

    @classmethod
    def resolve_active_articulator(
        cls,
        hand_landmarks: np.ndarray,
        articulator_type: str = "AUTO",
        anchor_3d: Optional[np.ndarray] = None,
        target_anchor_name: str = "NEUTRAL_SPACE"
    ) -> np.ndarray:
        """Determines active contacting point on hand, strictly enforcing fingertip contact for facial signs."""
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return np.array([0.50, 0.50, 0.0], dtype=np.float32)

        lm = np.array(hand_landmarks, dtype=np.float32)
        if lm.shape[-1] == 2:
            lm = np.pad(lm, ((0, 0), (0, 1)), mode="constant")

        art_type = articulator_type.upper()
        is_facial_anchor = target_anchor_name.upper() in [
            "CHIN", "LOWER_CHIN", "UPPER_LIP", "PHILTRUM", "LIP_UPPER",
            "NOSE", "FOREHEAD", "GLABELLA", "CHEEK", "CHEEK_RIGHT", "CHEEK_LEFT"
        ]

        if art_type in ["WRIST", "CARPAL"] and not is_facial_anchor:
            return lm[0]
        elif art_type in ["PALM_CENTER", "PALM"] and not is_facial_anchor:
            return (lm[0] + lm[9]) / 2.0
        elif art_type in ["INDEX_TIP", "POINT", "FINGERTIP_LEADER"]:
            return lm[8]
        elif art_type in ["THUMB_TIP"]:
            return lm[4]
        elif art_type in ["THUMB_INDEX_PINCH", "PINCH"]:
            return (lm[4] + lm[8]) / 2.0
        elif art_type in ["FINGERTIP", "FINGERTIPS", "FINGERTIPS_FLAT"]:
            return (lm[8] + lm[12]) / 2.0

        # AUTO mode: for all facial signs, strictly choose from fingertip contact candidates (NEVER wrist 0)
        if is_facial_anchor or (anchor_3d is not None and anchor_3d[1] < 0.52):
            candidates = [
                (lm[8] + lm[12]) / 2.0,  # Fingertips flat
                lm[8],                   # Index tip
                (lm[4] + lm[8]) / 2.0,   # Pinch center
                lm[4],                   # Thumb tip
                lm[12]                   # Middle tip
            ]
        else:
            candidates = [
                (lm[8] + lm[12]) / 2.0,
                (lm[0] + lm[9]) / 2.0,
                lm[0],
                lm[8],
                lm[4]
            ]

        if anchor_3d is not None:
            dists = [float(np.linalg.norm(c - anchor_3d)) for c in candidates]
            return candidates[int(np.argmin(dists))]

        return (lm[8] + lm[12]) / 2.0

    @classmethod
    def calculate_anchor_alignment(
        cls,
        hand_landmarks: np.ndarray,
        target_anchor_name: str,
        face_landmarks: Optional[np.ndarray] = None,
        pose_landmarks: Optional[np.ndarray] = None,
        articulator_type: str = "AUTO"
    ) -> Tuple[float, float, Optional[str], Dict[str, Any]]:
        """Calculates biometrically normalized fingertip-to-anchor alignment score and explicit Bengali hints."""
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return 0.0, 50.0, "⚠️ হাত ক্যামেরার সামনে প্রস্তুত রাখুন।", {}

        # 1. Resolve 3D Anchor & Fingertip-Centric Articulator
        anchor_3d = cls.get_anatomical_anchor_3d(target_anchor_name, face_landmarks, pose_landmarks)
        articulator_3d = cls.resolve_active_articulator(
            hand_landmarks,
            articulator_type,
            anchor_3d=anchor_3d,
            target_anchor_name=target_anchor_name
        )

        # 2. Biometric Scale Normalization (Face Height)
        face_height = 0.22
        if face_landmarks is not None and len(face_landmarks) >= 153:
            flm = np.array(face_landmarks, dtype=np.float32)
            fh = float(np.linalg.norm(flm[10] - flm[152]))
            if fh > 0.05:
                face_height = fh

        # 3. Euclidean 3D Distance & Biometric Ratio
        euclidean_dist = float(np.linalg.norm(articulator_3d - anchor_3d))
        dist_bio = euclidean_dist / face_height
        dist_cm = round(euclidean_dist * 50.0, 1)

        # Score computation:
        # Match <= 0.22 * H_face -> 92% - 100% score
        # 0.22 < dist_bio <= 0.45 -> 60% - 92% score
        # dist_bio > 0.45 -> drops to 0.0 at >= 0.80
        if dist_bio <= 0.22:
            score = 1.0 - (dist_bio / 0.22) * 0.08  # 0.92 - 1.00
        elif dist_bio <= 0.45:
            score = 0.92 - ((dist_bio - 0.22) / 0.23) * 0.32  # 0.60 - 0.92
        else:
            score = max(0.0, 0.60 - ((dist_bio - 0.45) / 0.35) * 0.60)  # 0.00 - 0.60

        # 4. Explicit Directional Guidance Vector mentioning 'আঙুলের ডগা'
        dx = float(articulator_3d[0] - anchor_3d[0])
        dy = float(articulator_3d[1] - anchor_3d[1])  # +y is downwards

        anchor_bn_map = {
            "CHIN": "চিবুকে",
            "LOWER_CHIN": "চিবুকে",
            "UPPER_LIP": "ঠোঁটের ওপর",
            "LIP_UPPER": "ঠোঁটের ওপর",
            "PHILTRUM": "ঠোঁটের ওপর",
            "NOSE": "নাকে",
            "CHEEK": "ডান গালে",
            "CHEEK_RIGHT": "ডান গালে",
            "CHEEK_LEFT": "বাম গালে",
            "FOREHEAD": "কপালে",
            "GLABELLA": "কপালে",
            "CHEST": "বুকের সামনে",
            "CHEST_MID": "বুকের মাঝে",
            "LEFT_WRIST": "বাম হাতের কবজিতে",
            "NEUTRAL_SPACE": "ক্যামেরা ফ্রেমের মাঝে"
        }
        loc_str = anchor_bn_map.get(target_anchor_name.upper(), "নির্দিষ্ট স্থানে")
        t_name = target_anchor_name.upper()

        hint = None
        if score < 0.75:
            # Special case 1: Forehead target but hand at chin (e.g. Salam)
            if t_name in ["FOREHEAD", "GLABELLA"] and articulator_3d[1] > 0.35:
                hint = f"⚠️ আঙুলের ডগা চিবুকে রয়েছে। আঙুলের ডগা উপরে {loc_str} স্পর্শ করুন।"
            # Special case 2: Chin target but hand at forehead (e.g. Dhonnobad)
            elif t_name in ["CHIN", "LOWER_CHIN", "UPPER_LIP"] and articulator_3d[1] < 0.28:
                hint = f"⚠️ আঙুলের ডগা কপালে রয়েছে। আঙুলের ডগা নামিয়ে {loc_str} স্পর্শ করুন।"
            # Special case 3: Hand at chest for face sign
            elif articulator_3d[1] >= 0.58 and t_name in ["CHIN", "UPPER_LIP", "NOSE", "FOREHEAD", "CHEEK", "CHEEK_RIGHT"]:
                hint = f"⚠️ আঙুলের ডগা বুকের কাছে। আঙুলের ডগা {loc_str} তুলুন।"
            elif dy > 0.10:
                hint = f"⚠️ আঙুলের ডগা নিচে রয়েছে। আঙুলের ডগা উপরে {loc_str} তুলুন।"
            elif dy < -0.10:
                hint = f"⚠️ আঙুলের ডগা অনেক উপরে রয়েছে। আঙুলের ডগা নামিয়ে {loc_str} স্পর্শ করুন।"
            elif dx > 0.08:
                hint = f"⚠️ আঙুলের ডগা একটু ডানে/বামে সমন্বয় করে {loc_str} স্পর্শ করুন।"
            elif dx < -0.08:
                hint = f"⚠️ আঙুলের ডগা একটু সমন্বয় করে {loc_str} স্পর্শ করুন।"
            else:
                hint = f"⚠️ আঙুলের ডগা আরও কাছে এনে {loc_str} স্পর্শ করুন।"

        debug_meta = {
            "anchor_3d": anchor_3d.tolist(),
            "articulator_3d": articulator_3d.tolist(),
            "dist_bio": dist_bio,
            "dist_cm": dist_cm,
            "face_height": face_height,
            "dx": dx,
            "dy": dy
        }

        return score, dist_cm, hint, debug_meta

    @classmethod
    def calculate_anchor_proximity(
        cls,
        wrist_lm: np.ndarray,
        target_anchor_name: str,
        face_lm: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Calculates normalized distance and cm distance using universal 3D anchor alignment."""
        if wrist_lm is None or len(wrist_lm) < 2:
            return 0.0, 50.0

        # Construct synthetic hand landmark array with wrist at position 0
        hand_lm = np.zeros((21, 3), dtype=np.float32)
        hand_lm[0, :len(wrist_lm)] = wrist_lm
        for i in range(1, 21):
            hand_lm[i, :len(wrist_lm)] = wrist_lm

        score, dist_cm, _, _ = cls.calculate_anchor_alignment(
            hand_landmarks=hand_lm,
            target_anchor_name=target_anchor_name,
            face_landmarks=face_lm,
            articulator_type="WRIST"
        )
        return score, dist_cm
