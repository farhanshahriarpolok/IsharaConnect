"""2D Kinematic Human Skeleton & Gesture Motion Sequence Interpolator.

Generates smooth 60-frame (30 FPS) upper-body kinematic animation loops and
articulates 21-landmark hand phalanges with inverse kinematics and natural micro-breathing motion.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("kinematic_interpolator")

# MediaPipe Hand Landmark Connections
HAND_CONNECTIONS = [
    # Palm Base
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)            # Palm Knuckle Bridge
]


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
    left_hand: List[Tuple[float, float]] = field(default_factory=list)   # 21 landmarks (x, y)
    right_hand: List[Tuple[float, float]] = field(default_factory=list)  # 21 landmarks (x, y)
    is_left_active: bool = False
    is_right_active: bool = True
    particle_trail: List[Tuple[float, float]] = field(default_factory=list)


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

        # 2. Synthesize smooth parametric Bezier / IK motion sequence
        return self._synthesize_parametric_sequence(slug)

    def _tensor_to_kinematic_frames(self, seq_tensor: np.ndarray, slug: str) -> List[KinematicJointFrame]:
        """Converts a (60, 151) dataset tensor into full upper-body kinematic frames."""
        frames = []
        is_dual = slug in ["dhonnobad", "sahajjo", "shagotom", "kemon_achen", "hospital", "bari"]
        
        for t in range(len(seq_tensor)):
            feat = seq_tensor[t]
            # Breathing motion
            breath = 0.005 * math.sin(2 * math.pi * t / 60.0)
            
            # Base upper body anchors in normalized [0, 1] rig space
            head = (0.50, 0.18 + breath * 0.5)
            neck = (0.50, 0.28 + breath * 0.7)
            chest = (0.50, 0.44 + breath)
            l_shoulder = (0.34, 0.32 + breath)
            r_shoulder = (0.66, 0.32 + breath)

            # Left and Right hand landmark slices
            l_lm = feat[0:63].reshape(21, 3)
            r_lm = feat[63:126].reshape(21, 3)

            has_left = np.any(np.abs(l_lm) > 1e-4) and is_dual
            has_right = np.any(np.abs(r_lm) > 1e-4) or not has_left

            # Right wrist and hand
            if has_right:
                # Map relative wrist motion to canvas signing zone
                rw_x = 0.62 + 0.12 * float(r_lm[0, 0])
                rw_y = 0.48 + 0.12 * float(r_lm[0, 1]) + breath
                r_wrist = (rw_x, rw_y)
                r_elbow = self._solve_ik_elbow(r_shoulder, r_wrist, is_right=True)
                r_hand = self._scale_hand_landmarks(r_lm, r_wrist, scale=0.08)
            else:
                r_wrist = (0.68, 0.76 + breath)
                r_elbow = (0.70, 0.55 + breath)
                r_hand = self._generate_default_hand(r_wrist, is_right=True, pose="rest")

            # Left wrist and hand
            if has_left:
                lw_x = 0.38 + 0.12 * float(l_lm[0, 0])
                lw_y = 0.48 + 0.12 * float(l_lm[0, 1]) + breath
                l_wrist = (lw_x, lw_y)
                l_elbow = self._solve_ik_elbow(l_shoulder, l_wrist, is_right=False)
                l_hand = self._scale_hand_landmarks(l_lm, l_wrist, scale=0.08)
            else:
                l_wrist = (0.32, 0.76 + breath)
                l_elbow = (0.30, 0.55 + breath)
                l_hand = self._generate_default_hand(l_wrist, is_right=False, pose="rest")

            frames.append(KinematicJointFrame(
                frame_idx=t,
                head=head,
                neck=neck,
                chest=chest,
                left_shoulder=l_shoulder,
                right_shoulder=r_shoulder,
                left_elbow=l_elbow,
                right_elbow=r_elbow,
                left_wrist=l_wrist,
                right_wrist=r_wrist,
                left_hand=l_hand,
                right_hand=r_hand,
                is_left_active=has_left,
                is_right_active=has_right,
                particle_trail=[r_wrist] if has_right else [l_wrist]
            ))

        return frames

    def _synthesize_parametric_sequence(self, slug: str) -> List[KinematicJointFrame]:
        """Synthesizes high-fidelity 60-frame natural motion loops with Bezier easing."""
        frames = []
        is_dual = slug in ["dhonnobad", "sahajjo", "shagotom", "kemon_achen", "hospital", "bari", "daktar"]

        # Trajectory definition per sign family
        traj_func = self._get_trajectory_function(slug)

        for t in range(60):
            # Phase factor in [0, 1]
            progress = t / 59.0
            breath = 0.004 * math.sin(2 * math.pi * progress)

            # Skeletal Core
            head = (0.50, 0.19 + breath * 0.5)
            neck = (0.50, 0.28 + breath * 0.7)
            chest = (0.50, 0.44 + breath)
            l_shoulder = (0.33, 0.32 + breath)
            r_shoulder = (0.67, 0.32 + breath)

            # Evaluate trajectory
            r_target, l_target, r_pose, l_pose, has_left = traj_func(progress)

            # Right Arm IK
            r_wrist = (r_target[0], r_target[1] + breath)
            r_elbow = self._solve_ik_elbow(r_shoulder, r_wrist, is_right=True)
            r_hand = self._generate_default_hand(r_wrist, is_right=True, pose=r_pose)

            # Left Arm IK
            if has_left:
                l_wrist = (l_target[0], l_target[1] + breath)
                l_elbow = self._solve_ik_elbow(l_shoulder, l_wrist, is_right=False)
                l_hand = self._generate_default_hand(l_wrist, is_right=False, pose=l_pose)
            else:
                l_wrist = (0.31, 0.75 + breath)
                l_elbow = (0.29, 0.54 + breath)
                l_hand = self._generate_default_hand(l_wrist, is_right=False, pose="rest")

            frames.append(KinematicJointFrame(
                frame_idx=t,
                head=head,
                neck=neck,
                chest=chest,
                left_shoulder=l_shoulder,
                right_shoulder=r_shoulder,
                left_elbow=l_elbow,
                right_elbow=r_elbow,
                left_wrist=l_wrist,
                right_wrist=r_wrist,
                left_hand=l_hand,
                right_hand=r_hand,
                is_left_active=has_left,
                is_right_active=True,
                particle_trail=[r_wrist]
            ))

        return frames

    def _get_trajectory_function(self, slug: str):
        """Returns trajectory generator for given sign."""
        if slug in ["dhonnobad", "thank_you"]:
            def traj(p: float):
                # Hand touches chin, extends outward/downward, then returns
                if p < 0.25:
                    # Move to chin
                    k = self._smooth_step(p / 0.25)
                    rw = (0.60 - 0.08 * k, 0.65 - 0.37 * k)
                    pose = "flat_open"
                elif p < 0.70:
                    # Extend outward forward
                    k = self._smooth_step((p - 0.25) / 0.45)
                    rw = (0.52 + 0.14 * k, 0.28 + 0.24 * k)
                    pose = "flat_open"
                else:
                    # Smooth loop back
                    k = self._smooth_step((p - 0.70) / 0.30)
                    rw = (0.66 - 0.06 * k, 0.52 + 0.13 * k)
                    pose = "flat_open"
                return rw, (0.31, 0.75), pose, "rest", False
            return traj

        elif slug in ["sahajjo", "help"]:
            def traj(p: float):
                # Left hand flat palm base, right hand on top lifting upward
                k = math.sin(math.pi * p)
                lw = (0.42, 0.62 - 0.06 * k)
                rw = (0.44, 0.54 - 0.12 * k)
                return rw, lw, "fist_on_palm", "flat_palm_up", True
            return traj

        elif slug in ["shagotom", "welcome"]:
            def traj(p: float):
                # Both hands sweep inward welcoming
                k = math.sin(math.pi * p)
                lw = (0.30 + 0.10 * k, 0.56 - 0.08 * k)
                rw = (0.70 - 0.10 * k, 0.56 - 0.08 * k)
                return rw, lw, "flat_open", "flat_open", True
            return traj

        elif slug in ["kemon_achen", "how_are_you"]:
            def traj(p: float):
                # Gentle double hand wave
                w = math.sin(4 * math.pi * p) * 0.04
                lw = (0.36 + w, 0.50)
                rw = (0.64 - w, 0.50)
                return rw, lw, "open_wave", "open_wave", True
            return traj

        elif slug in ["hospital", "daktar", "doctor"]:
            def traj(p: float):
                # Right fingers touch left wrist
                lw = (0.40, 0.60)
                k = math.sin(2 * math.pi * p) * 0.02
                rw = (0.42, 0.58 + k)
                return rw, lw, "index_touch", "flat_palm_up", True
            return traj

        elif slug in ["pani", "water", "khabar", "food"]:
            def traj(p: float):
                # Hand near mouth
                k = math.sin(math.pi * p)
                rw = (0.58 - 0.06 * k, 0.62 - 0.32 * k)
                return rw, (0.31, 0.75), "cup_hand", "rest", False
            return traj

        elif slug in ["ami", "me", "i"]:
            def traj(p: float):
                # Point to center chest
                k = math.sin(math.pi * p)
                rw = (0.62 - 0.12 * k, 0.60 - 0.16 * k)
                return rw, (0.31, 0.75), "point_inward", "rest", False
            return traj

        elif slug in ["thik_ache", "bhalo", "good", "ok"]:
            def traj(p: float):
                # Thumbs up gesture in chest area with micro pulse
                pulse = 0.02 * math.sin(2 * math.pi * p)
                rw = (0.60, 0.48 + pulse)
                return rw, (0.31, 0.75), "thumbs_up", "rest", False
            return traj

        else:
            # Default / Alphabet Static Articulation with Settling Wave
            def traj(p: float):
                settle = 0.015 * math.sin(2 * math.pi * p)
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

        # Base angle from shoulder to wrist
        base_angle = math.atan2(dy, dx)

        # Law of cosines for elbow interior angle
        cos_alpha = (l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha = math.acos(cos_alpha)

        # Outward bending convention
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
        """Generates 21 MediaPipe-style 2D hand landmark coordinates for standard poses."""
        wx, wy = wrist
        sign_x = 1.0 if is_right else -1.0
        landmarks = [(wx, wy)]  # 0: Wrist

        # Finger spread configuration angles (Thumb, Index, Middle, Ring, Pinky)
        finger_angles = [-35, -15, 0, 15, 30]
        lengths = [0.038, 0.048, 0.052, 0.047, 0.038]

        # Modify geometry based on pose
        curl_factors = [0.0, 0.0, 0.0, 0.0, 0.0]
        if pose == "thumbs_up":
            curl_factors = [-0.2, 0.85, 0.9, 0.9, 0.9]
        elif pose in ["point_inward", "index_touch"]:
            curl_factors = [0.8, 0.0, 0.9, 0.9, 0.9]
        elif pose in ["fist_on_palm", "rest"]:
            curl_factors = [0.6, 0.75, 0.8, 0.8, 0.75]
        elif pose == "alphabet_pose":
            curl_factors = [0.1, 0.1, 0.2, 0.2, 0.2]

        for f_idx in range(5):
            angle_rad = math.radians(finger_angles[f_idx] * sign_x - 90)
            f_len = lengths[f_idx]
            curl = curl_factors[f_idx]

            # 4 joints per finger (Base/MCP -> PIP -> DIP -> TIP)
            for j in range(1, 5):
                seg_len = (f_len / 4.0) * (1.0 - curl * 0.4)
                cur_dist = j * seg_len
                # Add curling deformation
                curl_offset_x = sign_x * curl * 0.015 * (j / 4.0)
                curl_offset_y = curl * 0.02 * (j / 4.0)

                jx = wx + cur_dist * math.cos(angle_rad) + curl_offset_x
                jy = wy + cur_dist * math.sin(angle_rad) + curl_offset_y
                landmarks.append((jx, jy))

        return landmarks

    def _scale_hand_landmarks(
        self,
        raw_lm: np.ndarray,
        wrist_pos: Tuple[float, float],
        scale: float = 0.08
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
    def _smooth_step(x: float) -> float:
        """Smooth Hermite cubic ease-in-out interpolation."""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)
