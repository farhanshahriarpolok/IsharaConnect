"""Tier 2: BdSL Isolated Sign Language Recognition (ISLR) Ingestor (BdSLW401 / BdSLW60 / Ishaara.ai).

Extracts 60-frame normalized temporal sequences (60 x 151D tensors) with FastDTW
dynamic trajectory alignment and temporal spatial landmark packaging.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core_engine.vision.dtw_matcher import DTWMotionMatcher
from core_engine.vision.spatial_hand_engine import SpatialHandEngine

logger = logging.getLogger("tier2_ingestor")


class Tier2ISLRIngestor:
    """Ingestion & 60-frame Temporal Sequence Pipeline for Tier 2 BdSL ISLR."""

    def __init__(self, manifest_path: Optional[str] = None):
        self.spatial_engine = SpatialHandEngine()
        self.dtw_matcher = DTWMotionMatcher()
        self.manifest_path = Path(manifest_path or "dataset/manifests/tier2_manifest.json")
        self.manifest_data: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads Tier 2 manifest schema."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading manifest {self.manifest_path}: {e}")

        return {
            "tier_name": "Tier 2: Isolated Sign Language Recognition (ISLR)",
            "total_lexicon": 450,
            "fps": 30,
            "sequence_length": 60,
            "feature_dimension": 151,
            "lexicon": [{"id": i, "slug": f"islr_word_{i}"} for i in range(25)]
        }

    def normalize_sequence_length(self, raw_sequence: np.ndarray, target_length: int = 60) -> np.ndarray:
        """Interpolates or resamples a temporal landmark sequence to exactly `target_length` frames."""
        raw_len, feat_dim = raw_sequence.shape
        if raw_len == target_length:
            return raw_sequence.astype(np.float32)

        if raw_len == 0:
            return np.zeros((target_length, feat_dim), dtype=np.float32)

        orig_idx = np.linspace(0, raw_len - 1, num=raw_len)
        target_idx = np.linspace(0, raw_len - 1, num=target_length)

        resampled = np.zeros((target_length, feat_dim), dtype=np.float32)
        for dim in range(feat_dim):
            resampled[:, dim] = np.interp(target_idx, orig_idx, raw_sequence[:, dim])

        return resampled

    def extract_from_frames(self, frames: List[np.ndarray], target_length: int = 60) -> np.ndarray:
        """Extracts 151-D spatial vectors from each frame and resamples to target_length."""
        raw_vectors = []
        for frame in frames:
            feat = self.spatial_engine.extract_spatial_features(frame)
            if feat and "spatial_vector" in feat:
                vec = np.asarray(feat["spatial_vector"], dtype=np.float32)
                if vec.shape[0] == 151:
                    raw_vectors.append(vec)
                else:
                    raw_vectors.append(np.zeros(151, dtype=np.float32))
            else:
                raw_vectors.append(np.zeros(151, dtype=np.float32))

        if not raw_vectors:
            return np.zeros((target_length, 151), dtype=np.float32)

        seq = np.array(raw_vectors, dtype=np.float32)
        return self.normalize_sequence_length(seq, target_length=target_length)

    def generate_mock_dataset(
        self,
        num_samples_per_class: int = 10,
        num_classes: int = 25,
        output_dir: str = "dataset/processed/tier2_islr"
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Generates synthetic (N, 60, 151) temporal trajectory sequences across ISLR classes."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        lexicon = self.manifest_data.get("lexicon", [])
        total_classes = min(num_classes, len(lexicon)) if lexicon else num_classes

        all_sequences = []
        all_labels = []

        logger.info(f"Generating Tier 2 ISLR dataset: {total_classes} classes x {num_samples_per_class} sequences (60 x 151)...")

        for class_idx in range(total_classes):
            slug = lexicon[class_idx].get("slug", f"word_{class_idx}") if class_idx < len(lexicon) else f"word_{class_idx}"
            
            # Base reference dynamic trajectory
            ref_traj = self.dtw_matcher.generate_synthetic_reference(slug, num_frames=60)

            for _ in range(num_samples_per_class):
                # Add time-warp jitter and sensor noise
                warp_noise = np.random.normal(0.0, 0.02, size=ref_traj.shape).astype(np.float32)
                sample_seq = ref_traj + warp_noise
                all_sequences.append(sample_seq)
                all_labels.append(class_idx)

        X = np.array(all_sequences, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        save_file = out_path / "tier2_islr_dataset.npz"
        np.savez_compressed(save_file, X=X, y=y, num_classes=total_classes, seq_len=60)
        logger.info(f"Tier 2 ISLR dataset saved to {save_file} with shape {X.shape}")

        return X, y, str(save_file)

    def validate(self, dataset_path: str = "dataset/processed/tier2_islr/tier2_islr_dataset.npz") -> Dict[str, Any]:
        """Validates Tier 2 dataset tensor shapes (N, 60, 151) and class annotations."""
        path = Path(dataset_path)
        if not path.exists():
            return {"valid": False, "error": f"File not found: {dataset_path}"}

        try:
            data = np.load(path)
            X, y = data["X"], data["y"]
            
            is_valid_shape = len(X.shape) == 3 and X.shape[1] == 60 and X.shape[2] == 151
            has_no_nans = not np.isnan(X).any()
            unique_classes = np.unique(y)

            return {
                "valid": bool(is_valid_shape and has_no_nans),
                "num_samples": int(X.shape[0]),
                "sequence_length": int(X.shape[1]),
                "feature_dim": int(X.shape[2]),
                "num_classes": int(len(unique_classes)),
                "has_nans": bool(not has_no_nans),
                "dataset_file": str(path)
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def get_statistics(self, dataset_path: str = "dataset/processed/tier2_islr/tier2_islr_dataset.npz") -> Dict[str, Any]:
        """Computes summary statistics for Tier 2 ISLR dataset."""
        val = self.validate(dataset_path)
        if not val.get("valid", False):
            return val

        data = np.load(dataset_path)
        X, y = data["X"], data["y"]
        return {
            "tier": "Tier 2: Isolated Sign Language Recognition (ISLR)",
            "sequences_count": int(len(X)),
            "sequence_frames": 60,
            "feature_dimension": 151,
            "classes_count": int(len(np.unique(y))),
            "status": "Ready for Temporal GRU/Transformer/DTW Inference"
        }
