"""2D Kinematic Human Skeleton & Gesture Motion Sequence Interpolator.

Generates smooth 60-frame (30 FPS) upper-body kinematic animation loops and
articulates 21-landmark hand phalanges with inverse kinematics, anatomically
accurate finger geometry, touch-contact detection, and natural micro-breathing.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("kinematic_interpolator")

# MediaPipe Hand Landmark Connections
HAND_CONNECTIONS = [
    # Palm Base
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm Knuckle Bridge
]

# Fingertip landmark indices: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

# Touch detection pairs (tip-to-tip proximity)
TOUCH_PAIRS = [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12)]
TOUCH_THRESHOLD = 0.022  # Normalized rig space


@dataclass
class KinematicJointFrame:
    """Represents the upper-body skeletal and dual-hand pose for a single animation frame."""
    frame_idx: int
    head: Tuple[float, float]
    neck: Tuple[float, float]
    chest: Tuple[float, float]
    left_shoulder: Tuple[float, float]
    right_shoulder: Tuple[float, float]
    left_elbow: Tuple[float, float]
    right_elbow: Tuple[float, float]
    left_wrist: Tuple[float, float]
    right_wrist: Tuple[float, float]
    left_hand: List[Tuple[float, float]] = field(default_factory=list)    # 21 landmarks (x, y)
    right_hand: List[Tuple[float, float]] = field(default_factory=list)   # 21 landmarks (x, y)
    is_left_active: bool = False
    is_right_active: bool = True
    particle_trail: List[Tuple[float, float]] = field(default_factory=list)
    # Touch contact pairs: (landmark_idx_a, landmark_idx_b, intensity [0.0-1.0])
    touch_contacts: List[Tuple[int, int, float]] = field(default_factory=list)
    # Z-depth for depth-sorted rendering: higher = closer to viewer (0.0–1.0)
    right_hand_z: float = 0.0
    left_hand_z: float = 0.0


class KinematicMotionInterpolator:
    """Resolves and synthesizes 60-frame kinematic human motion loops for any BdSL sign."""

    def __init__(
        self,
        tier2_dataset_path: str = "dataset/processed/tier2_islr/tier2_islr_dataset.npz",
        manifest_path: str = "dataset/manifests/tier2_manifest.json"
    ):
        self.tier2_dataset_path = Path(tier2_dataset_path)
        self.manifest_path = Path(manifest_path)
        self.dataset_X: Optional[np.ndarray] = None
        self.slug_to_index: Dict[str, int] = {}
        self._load_dataset()

    def _load_dataset(self):
        """Loads pre-compiled Tier 2 ISLR dataset tensors and manifest metadata if available."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for item in manifest.get("lexicon", []):
                    slug = item.get("slug", "").strip().lower()
                    item_id = item.get("id", -1)
                    if slug and item_id >= 0:
                        self.slug_to_index[slug] = item_id
            except Exception as e:
                logger.debug("Failed to parse tier2 manifest: %s", e)

        if self.tier2_dataset_path.exists():
            try:
                data = np.load(self.tier2_dataset_path, allow_pickle=True)
                if "X" in data:
                    self.dataset_X = data["X"]
                    logger.info("Loaded Tier 2 ISLR motion dataset: %s sequences.", len(self.dataset_X))
            except Exception as e:
                logger.debug("Could not load tier2_islr_dataset.npz: %s", e)

    def resolve_motion_sequence(
        self,
        sign_slug: str,
        label_bn: str = "",
        label_en: str = ""
    ) -> List[KinematicJointFrame]:
        """Resolves or generates a 60-frame kinematic animation loop for the given sign."""
        slug = sign_slug.strip().lower() if sign_slug else "dhonnobad"

        # 1. Check if direct dataset sequence exists
        if self.dataset_X is not None and slug in self.slug_to_index:
            idx = self.slug_to_index[slug]
            if 0 <= idx < len(self.dataset_X):
                seq_tensor = self.dataset_X[idx]  # Shape: (60, 151)
                return self._tensor_to_kinematic_frames(seq_tensor, slug)

        # 2. Synthesize smooth parametric motion sequence
        return self._synthesize_parametric_sequence(slug)

    def _tensor_to_kinematic_frames(self, seq_tensor: np.ndarray, slug: str) -> List[KinematicJointFrame]:
        """Converts a (60, 151) dataset tensor into full upper-body kinematic frames."""
        frames = []
        is_dual = slug in ["dhonnobad", "sahajjo", "shagotom", "kemon_achen", "hospital", "bari"]

        for t in range(len(seq_tensor)):
            feat = seq_tensor[t]
            breath = 0.005 * math.sin(2 * math.pi * t / 60.0)

            head = (0.50, 0.18 + breath * 0.5)
            neck = (0.50, 0.28 + breath * 0.7)
            chest = (0.50, 0.44 + breath)
            l_shoulder = (0.34, 0.32 + breath)
            r_shoulder = (0.66, 0.32 + breath)

            l_lm = feat[0:63].reshape(21, 3)
            r_lm = feat[63:126].reshape(21, 3)

            has_left = np.any(np.abs(l_lm) > 1e-4) and is_dual
            has_right = np.any(np.abs(r_lm) > 1e-4) or not has_left

            if has_right:
                rw_x = 0.62 + 0.14 * float(r_lm[0, 0])
                rw_y = 0.48 + 0.14 * float(r_lm[0, 1]) + breath
                r_wrist = (rw_x, rw_y)
                r_elbow = self._solve_ik_elbow(r_shoulder, r_wrist, is_right=True)
                r_hand = self._scale_hand_landmarks(r_lm, r_wrist, scale=0.14)
            else:
                r_wrist = (0.68, 0.76 + breath)
                r_elbow = (0.70, 0.55 + breath)
                r_hand = self._generate_default_hand(r_wrist, is_right=True, pose="rest")

            if has_left:
                lw_x = 0.38 + 0.14 * float(l_lm[0, 0])
                lw_y = 0.48 + 0.14 * float(l_lm[0, 1]) + breath
                l_wrist = (lw_x, lw_y)
                l_elbow = self._solve_ik_elbow(l_shoulder, l_wrist, is_right=False)
                l_hand = self._scale_hand_landmarks(l_lm, l_wrist, scale=0.14)
            else:
                l_wrist = (0.32, 0.76 + breath)
                l_elbow = (0.30, 0.55 + breath)
                l_hand = self._generate_default_hand(l_wrist, is_right=False, pose="rest")

            r_touches = self._compute_touch_contacts(r_hand) if has_right else []
            l_touches = self._compute_touch_contacts(l_hand) if has_left else []
            touch_contacts = r_touches + l_touches

            # Z-depth: higher wrist position (lower y) = closer to viewer
            r_z = max(0.0, min(1.0, 0.72 - r_wrist[1]))
            l_z = max(0.0, min(1.0, 0.72 - l_wrist[1]))

            frames.append(KinematicJointFrame(
                frame_idx=t,
                head=head, neck=neck, chest=chest,
                left_shoulder=l_shoulder, right_shoulder=r_shoulder,
                left_elbow=l_elbow, right_elbow=r_elbow,
                left_wrist=l_wrist, right_wrist=r_wrist,
                left_hand=l_hand, right_hand=r_hand,
                is_left_active=has_left, is_right_active=has_right,
                particle_trail=[r_wrist] if has_right else [l_wrist],
                touch_contacts=touch_contacts,
                right_hand_z=r_z, left_hand_z=l_z
            ))

        return frames

    def _synthesize_parametric_sequence(self, slug: str) -> List[KinematicJointFrame]:
        """Synthesizes high-fidelity 60-frame natural motion loops with Bezier easing."""
        frames = []
        is_dual = slug in [
            "dhonnobad", "sahajjo", "shagotom", "kemon_achen",
            "hospital", "bari", "daktar"
        ]

        traj_func = self._get_trajectory_function(slug)

        for t in range(60):
            progress = t / 59.0
            breath = 0.004 * math.sin(2 * math.pi * progress)

            head = (0.50, 0.19 + breath * 0.5)
            neck = (0.50, 0.28 + breath * 0.7)
            chest = (0.50, 0.44 + breath)
            l_shoulder = (0.33, 0.32 + breath)
            r_shoulder = (0.67, 0.32 + breath)

            r_target, l_target, r_pose, l_pose, has_left = traj_func(progress)

            r_wrist = (r_target[0], r_target[1] + breath)
            r_elbow = self._solve_ik_elbow(r_shoulder, r_wrist, is_right=True)
            r_hand = self._generate_default_hand(r_wrist, is_right=True, pose=r_pose)

            if has_left:
                l_wrist = (l_target[0], l_target[1] + breath)
                l_elbow = self._solve_ik_elbow(l_shoulder, l_wrist, is_right=False)
                l_hand = self._generate_default_hand(l_wrist, is_right=False, pose=l_pose)
            else:
                l_wrist = (0.31, 0.75 + breath)
                l_elbow = (0.29, 0.54 + breath)
                l_hand = self._generate_default_hand(l_wrist, is_right=False, pose="rest")

            r_touches = self._compute_touch_contacts(r_hand)
            l_touches = self._compute_touch_contacts(l_hand) if has_left else []

            r_z = max(0.0, min(1.0, 0.72 - r_wrist[1]))
            l_z = max(0.0, min(1.0, 0.72 - l_wrist[1]))

            frames.append(KinematicJointFrame(
                frame_idx=t,
                head=head, neck=neck, chest=chest,
                left_shoulder=l_shoulder, right_shoulder=r_shoulder,
                left_elbow=l_elbow, right_elbow=r_elbow,
                left_wrist=l_wrist, right_wrist=r_wrist,
                left_hand=l_hand, right_hand=r_hand,
                is_left_active=has_left, is_right_active=True,
                particle_trail=[r_wrist],
                touch_contacts=r_touches + l_touches,
                right_hand_z=r_z, left_hand_z=l_z
            ))

        return frames

    def _compute_touch_contacts(
        self,
        hand: List[Tuple[float, float]]
    ) -> List[Tuple[int, int, float]]:
        """Detects active fingertip touch pairs and returns (idxA, idxB, intensity) tuples."""
        contacts = []
        if len(hand) < 21:
            return contacts
        for a, b in TOUCH_PAIRS:
            ax, ay = hand[a]
            bx, by = hand[b]
            dist = math.hypot(ax - bx, ay - by)
            if dist < TOUCH_THRESHOLD:
                intensity = max(0.0, 1.0 - dist / TOUCH_THRESHOLD)
                contacts.append((a, b, intensity))
        return contacts

    def _get_trajectory_function(self, slug: str) -> Callable[[float], Any]:
        """Returns trajectory generator for given sign family."""

        if slug in ["dhonnobad", "thank_you"]:
            def traj(p: float):
                if p < 0.25:
                    k = self._smooth_step(p / 0.25)
                    rw = (0.60 - 0.08 * k, 0.65 - 0.37 * k)
                    pose = "flat_open"
                elif p < 0.70:
                    k = self._smooth_step((p - 0.25) / 0.45)
                    rw = (0.52 + 0.14 * k, 0.28 + 0.24 * k)
                    pose = "flat_open"
                else:
                    k = self._smooth_step((p - 0.70) / 0.30)
                    rw = (0.66 - 0.06 * k, 0.52 + 0.13 * k)
                    pose = "flat_open"
                return rw, (0.31, 0.75), pose, "rest", False
            return traj

        elif slug in ["sahajjo", "help"]:
            def traj(p: float):
                k = math.sin(math.pi * p)
                lw = (0.42, 0.62 - 0.06 * k)
                rw = (0.44, 0.54 - 0.12 * k)
                return rw, lw, "fist_closed", "flat_palm_up", True
            return traj

        elif slug in ["shagotom", "welcome"]:
            def traj(p: float):
                k = math.sin(math.pi * p)
                lw = (0.30 + 0.10 * k, 0.56 - 0.08 * k)
                rw = (0.70 - 0.10 * k, 0.56 - 0.08 * k)
                return rw, lw, "flat_open", "flat_open", True
            return traj

        elif slug in ["kemon_achen", "how_are_you"]:
            def traj(p: float):
                w = math.sin(4 * math.pi * p) * 0.04
                lw = (0.36 + w, 0.50)
                rw = (0.64 - w, 0.50)
                return rw, lw, "open_wave", "open_wave", True
            return traj

        elif slug in ["hospital", "daktar", "doctor"]:
            def traj(p: float):
                lw = (0.40, 0.60)
                k = math.sin(2 * math.pi * p) * 0.02
                rw = (0.42, 0.58 + k)
                return rw, lw, "index_point", "flat_palm_up", True
            return traj

        elif slug in ["pani", "water"]:
            def traj(p: float):
                k = math.sin(math.pi * p)
                rw = (0.58 - 0.06 * k, 0.62 - 0.32 * k)
                return rw, (0.31, 0.75), "cup_hand", "rest", False
            return traj

        elif slug in ["khabar", "food", "bhat"]:
            def traj(p: float):
                k = math.sin(2 * math.pi * p)
                rw = (0.60, 0.36 + 0.04 * k)
                return rw, (0.31, 0.75), "pinch_grip", "rest", False
            return traj

        elif slug in ["ami", "me", "i"]:
            def traj(p: float):
                k = math.sin(math.pi * p)
                rw = (0.62 - 0.12 * k, 0.60 - 0.16 * k)
                return rw, (0.31, 0.75), "index_point", "rest", False
            return traj

        elif slug in ["tumi", "you"]:
            def traj(p: float):
                k = self._smooth_step(math.sin(math.pi * p))
                rw = (0.64 - 0.18 * k, 0.52 - 0.08 * k)
                return rw, (0.31, 0.75), "index_point", "rest", False
            return traj

        elif slug in ["thik_ache", "bhalo", "good", "ok"]:
            def traj(p: float):
                pulse = 0.02 * math.sin(2 * math.pi * p)
                rw = (0.60, 0.48 + pulse)
                return rw, (0.31, 0.75), "thumbs_up", "rest", False
            return traj

        elif slug in ["na", "no"]:
            def traj(p: float):
                shake = math.sin(6 * math.pi * p) * 0.04
                rw = (0.62 + shake, 0.44)
                return rw, (0.31, 0.75), "index_point", "rest", False
            return traj

        elif slug in ["hae", "yes"]:
            def traj(p: float):
                nod = math.sin(4 * math.pi * p) * 0.025
                rw = (0.60, 0.46 + nod)
                return rw, (0.31, 0.75), "thumbs_up", "rest", False
            return traj

        elif slug in ["bari", "home", "house"]:
            def traj(p: float):
                k = math.sin(math.pi * p)
                lw = (0.40, 0.50 - 0.05 * k)
                rw = (0.60, 0.50 - 0.05 * k)
                return rw, lw, "index_point", "index_point", True
            return traj

        else:
            # Alphabet / generic static sign with micro settling wave
            def traj(p: float):
                settle = 0.012 * math.sin(2 * math.pi * p)
                rw = (0.61, 0.47 + settle)
                return rw, (0.31, 0.75), "alphabet_pose", "rest", False
            return traj

    def _solve_ik_elbow(
        self,
        shoulder: Tuple[float, float],
        wrist: Tuple[float, float],
        is_right: bool = True
    ) -> Tuple[float, float]:
        """2-Bone Inverse Kinematics solver returning natural elbow coordinate."""
        sx, sy = shoulder
        wx, wy = wrist
        l1, l2 = 0.18, 0.18

        dx = wx - sx
        dy = wy - sy
        dist = math.hypot(dx, dy)
        dist = max(0.01, min(l1 + l2 - 0.002, dist))

        base_angle = math.atan2(dy, dx)
        cos_alpha = (l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha = math.acos(cos_alpha)

        elbow_angle = base_angle + (alpha if is_right else -alpha)
        ex = sx + l1 * math.cos(elbow_angle)
        ey = sy + l1 * math.sin(elbow_angle)
        return (ex, ey)

    def _generate_default_hand(
        self,
        wrist: Tuple[float, float],
        is_right: bool = True,
        pose: str = "flat_open"
    ) -> List[Tuple[float, float]]:
        """Generates 21 MediaPipe-style 2D hand landmark coordinates with anatomical accuracy.

        Uses arc-based curling geometry: each joint bends inward (positive curl_deg)
        along a natural arc path, giving realistic proximal > distal phalange progression.

        Coordinate layout (normalized rig space, tip-up orientation):
            Landmark 0  = Wrist root
            Landmarks 1-4   = Thumb (CMC -> MCP -> IP -> TIP)
            Landmarks 5-8   = Index  (MCP -> PIP -> DIP -> TIP)
            Landmarks 9-12  = Middle (MCP -> PIP -> DIP -> TIP)
            Landmarks 13-16 = Ring   (MCP -> PIP -> DIP -> TIP)
            Landmarks 17-20 = Pinky  (MCP -> PIP -> DIP -> TIP)
        """
        wx, wy = wrist
        sign_x = 1.0 if is_right else -1.0
        landmarks: List[Tuple[float, float]] = [(wx, wy)]  # 0: Wrist root

        # --- Per-finger anatomical config ---
        # angle_base: direction from wrist to MCP (degrees from upward-vertical = -90°)
        # seg_lens: [proximal, intermediate, distal] segment lengths (normalized)
        # curl_deg: joint bend at [MCP, PIP, DIP] in degrees (0=straight, 90=fully curled)
        # spread_x: lateral offset factor at MCP

        POSE_MAP: Dict[str, List[Tuple[float, List[float], List[float], float]]] = {
            # (angle_base, seg_lens, curl_degs, spread_x) per finger [Thumb, Index, Middle, Ring, Pinky]
            "flat_open": [
                (-40, [0.032, 0.022, 0.018], [5, 5, 5],    -0.024 * sign_x),  # Thumb
                (-10, [0.038, 0.026, 0.020], [5, 5, 5],     0.000),           # Index
                (  0, [0.042, 0.028, 0.022], [5, 5, 5],     0.008 * sign_x),  # Middle
                ( 12, [0.038, 0.026, 0.020], [5, 5, 5],     0.016 * sign_x),  # Ring
                ( 24, [0.030, 0.020, 0.016], [5, 5, 5],     0.024 * sign_x),  # Pinky
            ],
            "flat_palm_up": [
                (-40, [0.032, 0.022, 0.018], [5, 5, 5],    -0.024 * sign_x),
                (-10, [0.038, 0.026, 0.020], [5, 5, 5],     0.000),
                (  0, [0.042, 0.028, 0.022], [5, 5, 5],     0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [5, 5, 5],     0.016 * sign_x),
                ( 24, [0.030, 0.020, 0.016], [5, 5, 5],     0.024 * sign_x),
            ],
            "open_wave": [
                (-42, [0.032, 0.022, 0.018], [8, 5, 5],    -0.022 * sign_x),
                (-12, [0.038, 0.026, 0.020], [8, 5, 5],     0.000),
                (  0, [0.042, 0.028, 0.022], [8, 5, 5],     0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [8, 5, 5],     0.016 * sign_x),
                ( 25, [0.030, 0.020, 0.016], [8, 5, 5],     0.024 * sign_x),
            ],
            "fist_closed": [
                (-30, [0.032, 0.022, 0.018], [45, 70, 70],  -0.016 * sign_x),  # Thumb over fist
                ( -8, [0.038, 0.026, 0.020], [80, 90, 85],   0.000),
                (  0, [0.042, 0.028, 0.022], [80, 90, 85],   0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [80, 90, 85],   0.016 * sign_x),
                ( 22, [0.030, 0.020, 0.016], [80, 85, 80],   0.022 * sign_x),
            ],
            "thumbs_up": [
                (-80, [0.032, 0.022, 0.018], [0, 0, 0],     -0.010 * sign_x),  # Thumb straight up
                ( -8, [0.038, 0.026, 0.020], [85, 90, 85],   0.000),
                (  0, [0.042, 0.028, 0.022], [85, 90, 85],   0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [85, 90, 85],   0.016 * sign_x),
                ( 22, [0.030, 0.020, 0.016], [85, 88, 80],   0.022 * sign_x),
            ],
            "index_point": [
                (-30, [0.032, 0.022, 0.018], [55, 70, 70],  -0.016 * sign_x),  # Thumb curled
                (-12, [0.038, 0.026, 0.020], [0, 0, 0],      0.000),            # Index extended
                (  0, [0.042, 0.028, 0.022], [80, 90, 85],   0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [80, 90, 85],   0.016 * sign_x),
                ( 22, [0.030, 0.020, 0.016], [80, 85, 80],   0.022 * sign_x),
            ],
            "pinch_grip": [
                (-40, [0.032, 0.022, 0.018], [20, 40, 55],  -0.018 * sign_x),  # Thumb pinch
                (-12, [0.038, 0.026, 0.020], [30, 55, 60],   0.000),            # Index pinch
                (  0, [0.042, 0.028, 0.022], [75, 85, 80],   0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [75, 85, 80],   0.016 * sign_x),
                ( 22, [0.030, 0.020, 0.016], [75, 82, 78],   0.022 * sign_x),
            ],
            "cup_hand": [
                (-38, [0.032, 0.022, 0.018], [35, 45, 40],  -0.020 * sign_x),
                (-10, [0.038, 0.026, 0.020], [40, 50, 45],   0.000),
                (  0, [0.042, 0.028, 0.022], [40, 50, 45],   0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [40, 50, 45],   0.016 * sign_x),
                ( 24, [0.030, 0.020, 0.016], [40, 48, 42],   0.024 * sign_x),
            ],
            "rest": [
                (-30, [0.032, 0.022, 0.018], [50, 65, 60],  -0.018 * sign_x),
                ( -8, [0.038, 0.026, 0.020], [70, 80, 75],   0.000),
                (  0, [0.042, 0.028, 0.022], [70, 80, 75],   0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [70, 80, 75],   0.016 * sign_x),
                ( 22, [0.030, 0.020, 0.016], [68, 78, 72],   0.022 * sign_x),
            ],
            "alphabet_pose": [
                (-38, [0.032, 0.022, 0.018], [12, 15, 12],  -0.022 * sign_x),
                (-10, [0.038, 0.026, 0.020], [12, 15, 12],   0.000),
                (  0, [0.042, 0.028, 0.022], [15, 18, 15],   0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [15, 18, 15],   0.016 * sign_x),
                ( 24, [0.030, 0.020, 0.016], [15, 18, 14],   0.024 * sign_x),
            ],
            "fist_on_palm": [
                (-30, [0.032, 0.022, 0.018], [45, 70, 70],  -0.016 * sign_x),
                ( -8, [0.038, 0.026, 0.020], [80, 90, 85],   0.000),
                (  0, [0.042, 0.028, 0.022], [80, 90, 85],   0.008 * sign_x),
                ( 12, [0.038, 0.026, 0.020], [80, 90, 85],   0.016 * sign_x),
                ( 22, [0.030, 0.020, 0.016], [80, 85, 80],   0.022 * sign_x),
            ],
        }

        finger_configs = POSE_MAP.get(pose, POSE_MAP["alphabet_pose"])

        for f_idx, config in enumerate(finger_configs):
            angle_base_deg, seg_lens, curl_degs, spread_x = config

            # Adjust direction for left hand (mirror)
            effective_angle = angle_base_deg * sign_x
            base_dir = math.radians(effective_angle - 90)  # -90 because canvas Y is downward

            # MCP (knuckle) position offset from wrist
            mcp_offset = 0.030 if f_idx > 0 else 0.024
            mcp_x = wx + spread_x + mcp_offset * math.cos(base_dir) * 0.5
            mcp_y = wy + mcp_offset * math.sin(base_dir)

            if f_idx == 0:
                # Thumb: starts from CMC offset on lateral side of wrist
                thumb_lateral = -0.018 * sign_x
                mcp_x = wx + thumb_lateral
                mcp_y = wy - 0.010

            landmarks.append((mcp_x, mcp_y))

            # Build phalange chain with arc-based curling
            cur_x, cur_y = mcp_x, mcp_y
            cur_angle = base_dir

            for seg_idx, (seg_len, curl_deg) in enumerate(zip(seg_lens, curl_degs)):
                # Convert curl_deg to additional rotation of this segment
                curl_rad = math.radians(curl_deg * sign_x)
                cur_angle = cur_angle + curl_rad * 0.35  # progressive cumulative bending

                end_x = cur_x + seg_len * math.cos(cur_angle)
                end_y = cur_y + seg_len * math.sin(cur_angle)
                landmarks.append((end_x, end_y))
                cur_x, cur_y = end_x, end_y

        return landmarks

    def _scale_hand_landmarks(
        self,
        raw_lm: np.ndarray,
        wrist_pos: Tuple[float, float],
        scale: float = 0.14
    ) -> List[Tuple[float, float]]:
        """Anchors and scales 21 normalized landmarks around current wrist position."""
        wx, wy = wrist_pos
        landmarks = []
        for i in range(21):
            lx = wx + float(raw_lm[i, 0]) * scale
            ly = wy + float(raw_lm[i, 1]) * scale
            landmarks.append((lx, ly))
        return landmarks

    @staticmethod
    def _catmull_rom_interp(
        p0: float, p1: float, p2: float, p3: float, t: float
    ) -> float:
        """Catmull-Rom spline interpolation for smooth continuous motion curves."""
        return 0.5 * (
            2 * p1
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
        )

    @staticmethod
    def _smooth_step(x: float) -> float:
        """Smooth Hermite cubic ease-in-out interpolation."""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def load_compound_motion(
        self,
        gloss_list: List[str]
    ) -> List[KinematicJointFrame]:
        """Synthesizes and stitches multi-sign sequential kinematic motion frames

        with smooth resting/co-articulation transitions between sub-signs.
        """
        if not gloss_list:
            return self.resolve_motion_sequence("dhonnobad")

        if len(gloss_list) == 1:
            return self.resolve_motion_sequence(gloss_list[0])

        compound_frames: List[KinematicJointFrame] = []
        transition_len = 10  # ~150ms transition buffer

        for sign_idx, gloss in enumerate(gloss_list):
            sign_frames = self.resolve_motion_sequence(gloss)

            for f in sign_frames:
                frame_copy = KinematicJointFrame(
                    frame_idx=len(compound_frames),
                    head=f.head, neck=f.neck, chest=f.chest,
                    left_shoulder=f.left_shoulder, right_shoulder=f.right_shoulder,
                    left_elbow=f.left_elbow, right_elbow=f.right_elbow,
                    left_wrist=f.left_wrist, right_wrist=f.right_wrist,
                    left_hand=list(f.left_hand), right_hand=list(f.right_hand),
                    is_left_active=f.is_left_active, is_right_active=f.is_right_active,
                    particle_trail=list(f.particle_trail),
                    touch_contacts=list(f.touch_contacts),
                    right_hand_z=f.right_hand_z, left_hand_z=f.left_hand_z
                )
                compound_frames.append(frame_copy)

            # Append resting transition between sub-signs
            if sign_idx < len(gloss_list) - 1:
                next_sign = gloss_list[sign_idx + 1]
                next_sign_frames = self.resolve_motion_sequence(next_sign)
                last_f = sign_frames[-1]
                first_next_f = next_sign_frames[0]

                for step in range(1, transition_len + 1):
                    alpha = step / float(transition_len + 1)
                    smooth_alpha = 0.5 * (1.0 - math.cos(math.pi * alpha))

                    def lerp(p1, p2):
                        return (p1[0] + (p2[0] - p1[0]) * smooth_alpha, p1[1] + (p2[1] - p1[1]) * smooth_alpha)

                    def lerp_hand(h1, h2):
                        if not h1 and not h2:
                            return []
                        if not h1:
                            return list(h2)
                        if not h2:
                            return list(h1)
                        return [lerp(h1[k], h2[k]) for k in range(min(len(h1), len(h2)))]

                    trans_frame = KinematicJointFrame(
                        frame_idx=len(compound_frames),
                        head=lerp(last_f.head, first_next_f.head),
                        neck=lerp(last_f.neck, first_next_f.neck),
                        chest=lerp(last_f.chest, first_next_f.chest),
                        left_shoulder=lerp(last_f.left_shoulder, first_next_f.left_shoulder),
                        right_shoulder=lerp(last_f.right_shoulder, first_next_f.right_shoulder),
                        left_elbow=lerp(last_f.left_elbow, first_next_f.left_elbow),
                        right_elbow=lerp(last_f.right_elbow, first_next_f.right_elbow),
                        left_wrist=lerp(last_f.left_wrist, first_next_f.left_wrist),
                        right_wrist=lerp(last_f.right_wrist, first_next_f.right_wrist),
                        left_hand=lerp_hand(last_f.left_hand, first_next_f.left_hand),
                        right_hand=lerp_hand(last_f.right_hand, first_next_f.right_hand),
                        is_left_active=last_f.is_left_active or first_next_f.is_left_active,
                        is_right_active=last_f.is_right_active or first_next_f.is_right_active,
                        particle_trail=[],
                        touch_contacts=[],
                        right_hand_z=last_f.right_hand_z + (first_next_f.right_hand_z - last_f.right_hand_z) * smooth_alpha,
                        left_hand_z=last_f.left_hand_z + (first_next_f.left_hand_z - last_f.left_hand_z) * smooth_alpha
                    )
                    compound_frames.append(trans_frame)

        return compound_frames

