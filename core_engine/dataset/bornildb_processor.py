"""BornilDB v1.0 Continuous Keypoint Matrix Extraction & Manifest Generator.

Extracts continuous 75-landmark normalized 3D coordinate matrices (T×75×3) from
BornilDB v1.0 corpus clips and produces aligned train/val/test manifests compatible
with the CSLROnnxEngine streaming pipeline.

Landmark node layout (75 nodes total):
  - Pose upper body (22 nodes):  MediaPipe Pose indices 0-21 (head → wrists)
  - Right hand (21 nodes):       MediaPipe Hands right side indices 0-20
  - Left hand  (21 nodes):       MediaPipe Hands left  side indices 0-20
  - Face contour (11 nodes):     MediaPipe FaceMesh keyframe subset (Forehead, Eyes, Nose, Lips, Chin)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed" / "bornildb"

# MediaPipe landmark group boundaries within the 75-node feature vector
LANDMARK_GROUPS = {
    "pose_upper": (0, 22),      # 22 upper-body pose nodes
    "right_hand": (22, 43),     # 21 right-hand nodes
    "left_hand":  (43, 64),     # 21 left-hand nodes
    "face_contour": (64, 75),   # 11 face keyframe nodes
}

# Canonical face keyframe node mapping (MediaPipe FaceMesh indices mapped to positions 64-74)
FACE_KEYFRAME_INDICES = {
    64: 10,   # Forehead glabella
    65: 152,  # Chin mental protuberance
    66: 0,    # Upper lip center
    67: 234,  # Right cheek
    68: 454,  # Left cheek
    69: 1,    # Nose tip
    70: 33,   # Right inner eye corner
    71: 133,  # Right outer eye corner
    72: 362,  # Left inner eye corner
    73: 263,  # Left outer eye corner
    74: 168,  # Bridge of nose
}


class LandmarkNormalizer:
    """Normalizes raw 75-node landmark sequences into scale-invariant 3D coordinate vectors."""

    def __init__(self, reference_node: int = 0, scale_ref_pair: Tuple[int, int] = (11, 12)):
        """
        Args:
            reference_node: Root anchor node for centering (default 0: nose/center).
            scale_ref_pair: Pair of nodes used for scale normalization (default 11,12: shoulders).
        """
        self.reference_node = reference_node
        self.scale_ref_pair = scale_ref_pair

    def normalize(self, landmarks_TN3: np.ndarray) -> np.ndarray:
        """Applies root-centering and inter-shoulder scale normalization.

        Args:
            landmarks_TN3: Raw landmark array of shape (T, N, 3)

        Returns:
            Normalized landmark array of shape (T, N, 3), scale-invariant.
        """
        if landmarks_TN3.ndim != 3 or landmarks_TN3.shape[1] < 22 or landmarks_TN3.shape[2] != 3:
            return landmarks_TN3.astype(np.float32)

        T, N, _ = landmarks_TN3.shape
        out = landmarks_TN3.astype(np.float32).copy()

        ref_l, ref_r = self.scale_ref_pair
        n_l = min(ref_l, N - 1)
        n_r = min(ref_r, N - 1)

        # Per-frame root centering and scale normalization
        for t in range(T):
            root = out[t, self.reference_node, :].copy()
            out[t] -= root

            # Inter-shoulder distance as scale denominator
            dist = float(np.linalg.norm(out[t, n_r, :2] - out[t, n_l, :2]))
            if dist > 1e-6:
                out[t] /= dist

        return out

    def batch_normalize(self, samples: List[np.ndarray]) -> List[np.ndarray]:
        """Normalizes a batch of variable-length landmark sequences."""
        return [self.normalize(s) for s in samples]


class BornilDBProcessor:
    """Processes BornilDB v1.0 manifests into normalized keypoint matrices for CSLR training."""

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        normalizer: Optional[LandmarkNormalizer] = None
    ):
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.normalizer = normalizer or LandmarkNormalizer()

    def load_manifest(self, split: str = "train") -> List[Dict[str, Any]]:
        """Loads a split manifest (train/val/test) from JSON."""
        manifest_path = self.processed_dir / f"manifest_{split}.json"
        if not manifest_path.exists():
            logger.warning("Manifest not found: %s. Returning empty list.", manifest_path)
            return []
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("samples", [])

    def load_keypoints(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        """Loads and normalizes the T×75×3 keypoint tensor for a single sample."""
        kp_str = sample.get("keypoints_path", "")
        if not kp_str:
            return None
        # Support both project-relative paths and absolute paths (e.g. pytest tmp dirs)
        kp_path = Path(kp_str)
        if not kp_path.is_absolute():
            kp_path = PROJECT_ROOT / kp_path
        if not kp_path.exists():
            logger.debug("Keypoints file not found: %s", kp_path)
            return None
        arr = np.load(str(kp_path)).astype(np.float32)
        if arr.ndim == 3 and arr.shape[1:] == (75, 3):
            return self.normalizer.normalize(arr)
        return arr


    def build_feature_window(self, keypoints_TN3: np.ndarray, window_size: int = 32) -> np.ndarray:
        """Resamples a variable-length keypoint sequence into a fixed window (32, 75, 3)."""
        T = keypoints_TN3.shape[0]
        if T == window_size:
            return keypoints_TN3
        # Linear interpolation for temporal resampling
        indices = np.linspace(0, T - 1, window_size)
        frames_int = indices.astype(int)
        frames_frac = (indices - frames_int).reshape(-1, 1, 1)
        frames_next = np.minimum(frames_int + 1, T - 1)
        resampled = keypoints_TN3[frames_int] * (1 - frames_frac) + keypoints_TN3[frames_next] * frames_frac
        return resampled.astype(np.float32)

    def extract_landmark_group(self, keypoints_TN3: np.ndarray, group: str) -> np.ndarray:
        """Extracts a named landmark group slice from a T×75×3 tensor."""
        start, end = LANDMARK_GROUPS.get(group, (0, 75))
        return keypoints_TN3[:, start:end, :]

    def compute_vocabulary_coverage(self, splits: List[str] = None) -> Dict[str, Any]:
        """Computes vocabulary coverage, class distribution, and sign transition statistics."""
        if splits is None:
            splits = ["train", "val", "test"]

        vocab_count: Dict[str, int] = {}
        transition_count: Dict[str, int] = {}
        total_samples = 0
        total_glosses = 0

        for split in splits:
            samples = self.load_manifest(split)
            for sample in samples:
                total_samples += 1
                seq = sample.get("gloss_sequence", [])
                total_glosses += len(seq)
                for gloss in seq:
                    vocab_count[gloss] = vocab_count.get(gloss, 0) + 1
                for i in range(len(seq) - 1):
                    key = f"{seq[i]}→{seq[i+1]}"
                    transition_count[key] = transition_count.get(key, 0) + 1

        # Sort by frequency
        vocab_sorted = sorted(vocab_count.items(), key=lambda x: x[1], reverse=True)
        top_transitions = sorted(transition_count.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "total_samples": total_samples,
            "total_glosses": total_glosses,
            "vocabulary_size": len(vocab_count),
            "top_30_glosses": vocab_sorted[:30],
            "class_distribution": dict(vocab_sorted),
            "top_20_transitions": dict(top_transitions),
            "avg_glosses_per_sentence": total_glosses / max(total_samples, 1)
        }

    def generate_all_manifests(self, max_per_split: int = 500) -> Dict[str, int]:
        """Regenerates all manifests and returns sample counts per split."""
        from scripts.download_and_ingest_bornildb import BornilDBIngestor
        ingestor = BornilDBIngestor(
            output_dir=self.processed_dir,
            samples=max_per_split
        )
        stats = ingestor.ingest()
        return stats.get("split_counts", {})
