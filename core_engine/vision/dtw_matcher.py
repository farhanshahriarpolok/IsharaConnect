"""Dynamic Time Warping (DTW) Trajectory & Motion Matcher for BdSL Dynamic Signs.

Evaluates spatial-temporal trajectory alignment between live user gesture buffers
and reference dynamic signs (e.g. 'dhonnobad', 'kemon_achen', 'sahajjo').
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DTWMotionMatcher:
    """Dynamic Time Warping (DTW) Motion & Gesture Alignment Engine."""

    def __init__(self, distance_scale: float = 8.0, match_threshold: float = 12.0):
        """Initializes the DTW matcher.

        Args:
            distance_scale: Sensitivity temperature factor for exponential score mapping.
            match_threshold: Maximum normalized distance to consider a gesture a valid match.
        """
        self.distance_scale = distance_scale
        self.match_threshold = match_threshold
        self._reference_cache: Dict[str, np.ndarray] = {}
        self._init_built_in_references()

    def _init_built_in_references(self):
        """Pre-seeds standard BdSL dynamic sign reference trajectories."""
        # Built-in reference trajectories of shape (30, 151)
        for sign_name in ["dhonnobad", "kemon_achen", "sahajjo", "shagotom", "shobaike_dhonnobad"]:
            self._reference_cache[sign_name] = self.generate_synthetic_reference(sign_name)

    @staticmethod
    def compute_pairwise_distances(seq1: np.ndarray, seq2: np.ndarray) -> np.ndarray:
        """Computes Euclidean distance matrix between every pair of frames in seq1 and seq2.

        Args:
            seq1: Array of shape (T1, D)
            seq2: Array of shape (T2, D)

        Returns:
            Cost matrix of shape (T1, T2)
        """
        # Vectorized Euclidean distance matrix: ||a - b||_2
        diff = seq1[:, np.newaxis, :] - seq2[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=-1)

    def compute_dtw_distance(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray
    ) -> Tuple[float, float, List[Tuple[int, int]], np.ndarray]:
        """Calculates DTW distance, normalized path distance, optimal warping path, and cost matrix.

        Args:
            seq1: Test sequence of shape (T1, D)
            seq2: Reference sequence of shape (T2, D)

        Returns:
            Tuple of:
                - total_cost (float): Total cumulative warping cost
                - normalized_cost (float): total_cost / path_length
                - path (List[Tuple[int, int]]): Coordinates of the optimal warping path
                - dtw_matrix (np.ndarray): Full cumulative cost matrix
        """
        s1 = np.asarray(seq1, dtype=np.float32)
        s2 = np.asarray(seq2, dtype=np.float32)

        if s1.ndim == 1:
            s1 = s1.reshape(1, -1)
        if s2.ndim == 1:
            s2 = s2.reshape(1, -1)

        t1, d1 = s1.shape
        t2, d2 = s2.shape

        if d1 != d2:
            min_d = min(d1, d2)
            s1 = s1[:, :min_d]
            s2 = s2[:, :min_d]

        # 1. Compute pairwise frame distance matrix
        cost_matrix = self.compute_pairwise_distances(s1, s2)

        # 2. Dynamic programming cost matrix
        dtw_matrix = np.full((t1, t2), np.inf, dtype=np.float32)
        dtw_matrix[0, 0] = cost_matrix[0, 0]

        # First column
        for i in range(1, t1):
            dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + cost_matrix[i, 0]

        # First row
        for j in range(1, t2):
            dtw_matrix[0, j] = dtw_matrix[0, j - 1] + cost_matrix[0, j]

        # Fill table
        for i in range(1, t1):
            for j in range(1, t2):
                min_prev = min(
                    dtw_matrix[i - 1, j],      # Insertion
                    dtw_matrix[i, j - 1],      # Deletion
                    dtw_matrix[i - 1, j - 1]   # Match
                )
                dtw_matrix[i, j] = cost_matrix[i, j] + min_prev

        # 3. Backtrack to find optimal warping path
        path = [(t1 - 1, t2 - 1)]
        i, j = t1 - 1, t2 - 1
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = [
                    (dtw_matrix[i - 1, j - 1], (i - 1, j - 1)),
                    (dtw_matrix[i - 1, j], (i - 1, j)),
                    (dtw_matrix[i, j - 1], (i, j - 1))
                ]
                _, next_step = min(candidates, key=lambda x: x[0])
                i, j = next_step
            path.append((i, j))

        path.reverse()
        total_cost = float(dtw_matrix[t1 - 1, t2 - 1])
        path_len = len(path)
        normalized_cost = total_cost / max(1, path_len)

        return total_cost, normalized_cost, path, dtw_matrix

    def evaluate_gesture_accuracy(
        self,
        user_frames: np.ndarray,
        reference_frames: Union[np.ndarray, str],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluates user gesture accuracy against reference sign sequence.

        Args:
            user_frames: Live gesture trajectory (T1, 151).
            reference_frames: Reference trajectory (T2, 151) or sign label name.
            threshold: Optional custom match threshold.

        Returns:
            Dictionary containing:
                - distance (float)
                - normalized_distance (float)
                - score (float): 0.0 to 100.0%
                - match_percentage (float): Alias for score
                - is_match (bool)
                - path_length (int)
                - warping_path (List[Tuple[int, int]])
        """
        thresh = threshold if threshold is not None else self.match_threshold

        if isinstance(reference_frames, str):
            ref_seq = self.get_reference_trajectory(reference_frames)
        else:
            ref_seq = np.asarray(reference_frames, dtype=np.float32)

        user_seq = np.asarray(user_frames, dtype=np.float32)

        if len(user_seq) == 0 or len(ref_seq) == 0:
            return {
                "distance": 0.0,
                "normalized_distance": 0.0,
                "score": 0.0,
                "match_percentage": 0.0,
                "is_match": False,
                "path_length": 0,
                "warping_path": []
            }

        total_cost, norm_cost, path, _ = self.compute_dtw_distance(user_seq, ref_seq)

        # Smooth exponential score mapping: Score = 100 * exp(-norm_cost / distance_scale)
        score_exp = 100.0 * np.exp(-norm_cost / max(0.1, self.distance_scale))
        match_score = float(np.clip(score_exp, 0.0, 100.0))
        is_match = bool(norm_cost <= thresh and match_score >= 50.0)

        return {
            "distance": round(total_cost, 4),
            "normalized_distance": round(norm_cost, 4),
            "score": round(match_score, 2),
            "match_percentage": round(match_score, 2),
            "is_match": is_match,
            "path_length": len(path),
            "warping_path": path
        }

    def match_sign(
        self,
        user_sequence: np.ndarray,
        candidates: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Classifies user dynamic gesture against a dictionary of candidate reference trajectories.

        Args:
            user_sequence: Live gesture frames (T, 151).
            candidates: Dict of sign_name -> reference sequence (T_ref, 151).
                        If None, uses cached reference library.

        Returns:
            Dict containing best_match_sign, best_score, best_distance, and all ranked candidates.
        """
        cand_dict = candidates if candidates is not None else self._reference_cache
        if not cand_dict:
            return {"best_match": None, "best_score": 0.0, "rankings": []}

        results = []
        for sign_name, ref_seq in cand_dict.items():
            eval_res = self.evaluate_gesture_accuracy(user_sequence, ref_seq)
            results.append({
                "sign": sign_name,
                "score": eval_res["score"],
                "normalized_distance": eval_res["normalized_distance"],
                "is_match": eval_res["is_match"]
            })

        results.sort(key=lambda x: x["normalized_distance"])
        best = results[0] if results else None

        return {
            "best_match": best["sign"] if best else None,
            "best_score": best["score"] if best else 0.0,
            "is_match": best["is_match"] if best else False,
            "rankings": results
        }

    def register_reference(self, sign_name: str, sequence: np.ndarray):
        """Registers or overrides a reference dynamic sequence in memory."""
        self._reference_cache[sign_name] = np.asarray(sequence, dtype=np.float32)

    def get_reference_trajectory(self, sign_name: str) -> np.ndarray:
        """Gets reference trajectory from cache or generates synthetic one."""
        if sign_name in self._reference_cache:
            return self._reference_cache[sign_name]
        generated = self.generate_synthetic_reference(sign_name)
        self._reference_cache[sign_name] = generated
        return generated

    @staticmethod
    def generate_synthetic_reference(sign_name: str, num_frames: int = 30) -> np.ndarray:
        """Generates a realistic smooth (num_frames, 151) spatial-temporal reference trajectory.

        Args:
            sign_name: Name/slug of the dynamic sign.
            num_frames: Length of sequence (default 30).

        Returns:
            np.ndarray of shape (num_frames, 151).
        """
        t = np.linspace(0.0, np.pi, num_frames, dtype=np.float32)
        ref = np.zeros((num_frames, 151), dtype=np.float32)

        # Base motion trajectories depending on semantic gesture type
        if "dhonnobad" in sign_name:
            # Outward forward wave (wrist z movement and y descent)
            for i in range(num_frames):
                # Right wrist (idx 21*3 = 63, 64, 65)
                ref[i, 63] = 0.2 + 0.1 * np.sin(t[i])      # X
                ref[i, 64] = -0.1 + 0.2 * np.cos(t[i])     # Y
                ref[i, 65] = -0.3 * np.sin(t[i])           # Z (forward reach)
                # Touch matrix for open palm (no fingertip contact)
                ref[i, 126:] = 5.0
        elif "kemon" in sign_name:
            # Side to side oscillation
            for i in range(num_frames):
                ref[i, 63] = 0.3 * np.sin(2 * t[i])
                ref[i, 64] = -0.1 * np.cos(t[i])
                ref[i, 126:] = 6.0
        elif "sahajjo" in sign_name:
            # Upward dual-hand support movement
            for i in range(num_frames):
                # Left palm base
                ref[i, 0] = -0.15
                ref[i, 1] = 0.2 - 0.3 * np.sin(t[i])
                # Right fist on top
                ref[i, 63] = -0.15
                ref[i, 64] = 0.1 - 0.3 * np.sin(t[i])
                # Close contact matrix distance (contact active)
                ref[i, 126:151] = 0.5 * (1.0 - np.sin(t[i]))
        else:
            # Generic smooth gesture arc
            for i in range(num_frames):
                ref[i, 0] = -0.2 * np.cos(t[i])
                ref[i, 63] = 0.2 * np.sin(t[i])
                ref[i, 126:] = 4.0

        return ref
