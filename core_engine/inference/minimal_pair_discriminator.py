"""Fine-Grained Minimal Pair Discriminator for Ambiguous BdSL Gestures.

Resolves visually similar sign pairs using kinematic phase correlation,
frequency domain analysis, trajectory integration, and anatomical anchor point proximity:
1. ভূমিকম্প (Earthquake) vs. যানজট (Traffic Jam)
2. চাচা (Uncle) vs. দাদা/নানা (Grandfather)
3. দেবর (Brother-in-law) vs. দুলাভাই (Elder Brother-in-law)
4. তাড়াতাড়ি (Hurry) vs. ছুড়ে মারা (Throw)
5. বাবা (Father) vs. মা (Mother)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Known minimal pair clusters
MINIMAL_PAIR_CLUSTERS = {
    "earthquake_traffic": ["bhumikompo", "janjot", "ভূমিকম্প", "যানজট"],
    "uncle_grandfather": ["chacha", "dada", "nana", "চাচা", "দাদা", "নানা"],
    "in_laws_brother": ["debor", "dulabhai", "দেবর", "দুলাভাই"],
    "hurry_throw": ["taratari", "chure_mara", "তাড়াতাড়ি", "ছুড়ে মারা"],
    "father_mother": ["baba", "ma", "বাবা", "মা"],
}


class MinimalPairDiscriminator:
    """Disambiguates visually ambiguous minimal pairs using spatio-temporal kinematic features."""

    def __init__(self):
        pass

    def identify_cluster(self, candidates: List[str]) -> Optional[str]:
        """Identifies if any pair of candidate signs forms a known minimal pair cluster."""
        cand_set = {c.strip().lower() for c in candidates if c}
        for cluster_name, items in MINIMAL_PAIR_CLUSTERS.items():
            item_set = {it.lower() for it in items}
            if len(cand_set.intersection(item_set)) >= 1:
                return cluster_name
        return None

    def disambiguate_earthquake_vs_traffic(
        self,
        left_traj: Optional[np.ndarray],
        right_traj: Optional[np.ndarray],
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """Differentiates Earthquake (in-phase lateral vibration) vs Traffic Jam (alternating translation).

        Args:
            left_traj: (T, 3) 3D coordinate sequence of left wrist/hand.
            right_traj: (T, 3) 3D coordinate sequence of right wrist/hand.
        """
        if left_traj is None or right_traj is None or len(left_traj) < 6 or len(right_traj) < 6:
            return {
                "resolved_slug": "bhumikompo",
                "resolved_bn": "ভূমিকম্প",
                "confidence": 0.80,
                "rationale": "Default fallback due to short trajectory buffer."
            }

        # Check synchronous lateral (x-axis) vs alternating forward-backward (z-axis or y-axis)
        min_len = min(len(left_traj), len(right_traj))
        lx = left_traj[-min_len:, 0]
        rx = right_traj[-min_len:, 0]
        lz = left_traj[-min_len:, 2] if left_traj.shape[1] > 2 else left_traj[-min_len:, 1]
        rz = right_traj[-min_len:, 2] if right_traj.shape[1] > 2 else right_traj[-min_len:, 1]

        # Normalized cross-correlation
        lx_norm = (lx - np.mean(lx)) / (np.std(lx) + 1e-6)
        rx_norm = (rx - np.mean(rx)) / (np.std(rx) + 1e-6)
        corr_x = float(np.mean(lx_norm * rx_norm))

        lz_norm = (lz - np.mean(lz)) / (np.std(lz) + 1e-6)
        rz_norm = (rz - np.mean(rz)) / (np.std(rz) + 1e-6)
        corr_z = float(np.mean(lz_norm * rz_norm))

        # Frequency / zero crossings of lateral vibration
        diffs = np.diff(rx)
        zero_crossings = np.sum(diffs[:-1] * diffs[1:] < 0)
        vibration_freq = (zero_crossings / 2.0) * (fps / len(rx))

        # Earthquake is characterized by synchronous vibration (corr_x > 0.3, freq >= 3.0 Hz)
        if corr_x > 0.25 or (vibration_freq >= 2.5 and corr_z > -0.3):
            return {
                "resolved_slug": "bhumikompo",
                "resolved_bn": "ভূমিকম্প",
                "confidence": max(0.85, min(0.98, float(0.70 + vibration_freq * 0.05))),
                "rationale": f"In-phase lateral vibration detected (freq={vibration_freq:.1f}Hz, corr_x={corr_x:.2f})."
            }
        else:
            return {
                "resolved_slug": "janjot",
                "resolved_bn": "যানজট",
                "confidence": 0.90,
                "rationale": f"Out-of-phase alternating longitudinal motion detected (corr_z={corr_z:.2f})."
            }

    def disambiguate_uncle_vs_grandfather(
        self,
        trajectory: np.ndarray,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """Differentiates Uncle (acute chin touch) vs Grandfather (inferior beard stroke)."""
        if trajectory is None or len(trajectory) < 4:
            return {
                "resolved_slug": "chacha",
                "resolved_bn": "চাচা",
                "confidence": 0.75,
                "rationale": "Default acute chin touch fallback."
            }

        # Downward vertical displacement (y-axis increase in image coordinates or decrease in 3D world)
        y_coords = trajectory[:, 1]
        delta_y = float(np.max(y_coords) - np.min(y_coords))

        # Terminal velocity
        velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1) * fps
        terminal_vel = float(velocities[-1]) if len(velocities) > 0 else 0.0

        # Grandfather has extended downward stroke (delta_y > 0.08 normalized or sustained stroke)
        if delta_y >= 0.075:
            return {
                "resolved_slug": "dada",
                "resolved_bn": "দাদা",
                "confidence": 0.92,
                "rationale": f"Extended downward beard stroke detected (delta_y={delta_y:.3f})."
            }
        else:
            return {
                "resolved_slug": "chacha",
                "resolved_bn": "চাচা",
                "confidence": 0.89,
                "rationale": f"Acute stationary chin tap detected (delta_y={delta_y:.3f}, v_term={terminal_vel:.2f})."
            }

    def disambiguate_debor_vs_dulabhai(
        self,
        landmarks: np.ndarray
    ) -> Dict[str, Any]:
        """Differentiates Debor (1 index digit) vs Dulabhai (2 digits V-shape over nose).

        Args:
            landmarks: (21, 3) single active hand landmarks.
        """
        if landmarks is None or len(landmarks) < 21:
            return {
                "resolved_slug": "debor",
                "resolved_bn": "দেবর",
                "confidence": 0.75,
                "rationale": "Default single digit fallback."
            }

        # Index tip (8), Middle tip (12), Index PIP (6), Middle PIP (10), Wrist (0)
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        wrist = landmarks[0]

        index_ext = float(np.linalg.norm(index_tip - wrist))
        middle_ext = float(np.linalg.norm(middle_tip - wrist))
        tip_dist = float(np.linalg.norm(index_tip - middle_tip))

        # If both index and middle are extended and separated (V-shape)
        is_v_shape = middle_ext >= 0.70 * index_ext and tip_dist >= 0.045

        if is_v_shape:
            return {
                "resolved_slug": "dulabhai",
                "resolved_bn": "দুলাভাই",
                "confidence": 0.93,
                "rationale": f"V-shape dual-digit extension detected (tip_dist={tip_dist:.3f})."
            }
        else:
            return {
                "resolved_slug": "debor",
                "resolved_bn": "দেবর",
                "confidence": 0.91,
                "rationale": f"Single index finger touch detected (middle_ext_ratio={middle_ext/max(1e-6, index_ext):.2f})."
            }

    def disambiguate_hurry_vs_throw(
        self,
        trajectory: np.ndarray,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """Differentiates Hurry (vertical flutter) vs Throw (anterior forward thrust)."""
        if trajectory is None or len(trajectory) < 4:
            return {
                "resolved_slug": "taratari",
                "resolved_bn": "তাড়াতাড়ি",
                "confidence": 0.80,
                "rationale": "Default flutter fallback."
            }

        # Compute accelerations
        velocities = np.diff(trajectory, axis=0) * fps
        accels = np.diff(velocities, axis=0) * fps if len(velocities) >= 2 else velocities

        # Vertical (y) vs Anterior (z) acceleration peak
        max_ay = float(np.max(np.abs(accels[:, 1]))) if len(accels) > 0 else 0.0
        max_az = float(np.max(np.abs(accels[:, 2]))) if accels.shape[1] > 2 else 0.0

        # Net forward displacement
        net_z = float(trajectory[-1, 2] - trajectory[0, 2]) if trajectory.shape[1] > 2 else 0.0

        if net_z > 0.15 or max_az > (max_ay * 1.2):
            return {
                "resolved_slug": "chure_mara",
                "resolved_bn": "ছুড়ে মারা",
                "confidence": 0.91,
                "rationale": f"Anterior ballistic forward thrust detected (net_z={net_z:.2f}, max_az={max_az:.1f})."
            }
        else:
            return {
                "resolved_slug": "taratari",
                "resolved_bn": "তাড়াতাড়ি",
                "confidence": 0.93,
                "rationale": f"High-acceleration vertical flutter detected (max_ay={max_ay:.1f})."
            }

    def disambiguate_father_vs_mother(
        self,
        landmarks: np.ndarray,
        trajectory: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Differentiates Father (upper lip mustache swipe) vs Mother (cheek double tap)."""
        if landmarks is None or len(landmarks) < 21:
            return {
                "resolved_slug": "baba",
                "resolved_bn": "বাবা",
                "confidence": 0.75,
                "rationale": "Default kinship fallback."
            }

        index_tip = landmarks[8]
        wrist = landmarks[0]

        # Horizontal swipe displacement in trajectory
        has_horizontal_swipe = False
        if trajectory is not None and len(trajectory) >= 4:
            x_range = float(np.max(trajectory[:, 0]) - np.min(trajectory[:, 0]))
            if x_range >= 0.06:
                has_horizontal_swipe = True

        # Philtrum / Mustache zone vs Cheek lateral position
        # Index tip x-coord relative to center (0.5 in normalized coordinates)
        x_offset = abs(float(index_tip[0]) - 0.5)

        if has_horizontal_swipe or x_offset < 0.10:
            return {
                "resolved_slug": "baba",
                "resolved_bn": "বাবা",
                "confidence": 0.92,
                "rationale": "Philtrum upper-lip mustache swipe motion detected."
            }
        else:
            return {
                "resolved_slug": "ma",
                "resolved_bn": "মা",
                "confidence": 0.90,
                "rationale": "Lateral cheek double-tap anchor detected."
            }

    def disambiguate(
        self,
        candidate_slug: str,
        trajectory_3d: Optional[np.ndarray] = None,
        left_landmarks: Optional[np.ndarray] = None,
        right_landmarks: Optional[np.ndarray] = None,
        fps: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Unified entrypoint for minimal pair disambiguation."""
        cand = candidate_slug.strip().lower()

        # 1. Earthquake vs Traffic
        if cand in ["bhumikompo", "janjot", "ভূমিকম্প", "যানজট"]:
            return self.disambiguate_earthquake_vs_traffic(trajectory_3d, trajectory_3d, fps=fps)

        # 2. Uncle vs Grandfather
        if cand in ["chacha", "dada", "nana", "চাচা", "দাদা", "নানা"]:
            traj = trajectory_3d if trajectory_3d is not None else (right_landmarks if right_landmarks is not None else left_landmarks)
            if traj is not None:
                return self.disambiguate_uncle_vs_grandfather(traj, fps=fps)

        # 3. Debor vs Dulabhai
        if cand in ["debor", "dulabhai", "দেবর", "দুলাভাই"]:
            lm = right_landmarks if right_landmarks is not None else left_landmarks
            if lm is not None:
                return self.disambiguate_debor_vs_dulabhai(lm)

        # 4. Hurry vs Throw
        if cand in ["taratari", "chure_mara", "তাড়াতাড়ি", "ছুড়ে মারা"]:
            if trajectory_3d is not None:
                return self.disambiguate_hurry_vs_throw(trajectory_3d, fps=fps)

        # 5. Father vs Mother
        if cand in ["baba", "ma", "বাবা", "মা"]:
            lm = right_landmarks if right_landmarks is not None else left_landmarks
            if lm is not None:
                return self.disambiguate_father_vs_mother(lm, trajectory_3d)

        return None
