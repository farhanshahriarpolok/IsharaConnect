"""151-D Spatial Feature Extractor & Augmentation Pipeline.

Extracts normalized 126-D landmark coordinates and 25-D fingertip touch distance matrices
(151-D vector) from dataset/raw_samples/ and saves into dataset/spatial_landmarks/.
"""

import glob
import logging
import os
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from core_engine.preprocessing.normalizer import LandmarkNormalizer
from core_engine.vision.spatial_hand_engine import SpatialHandEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_SAMPLES_DIR = Path("dataset/raw_samples")
OUTPUT_DIR = Path("dataset/spatial_landmarks")


def extract_151d_from_landmarks(
    left_lm: Optional[np.ndarray],
    right_lm: Optional[np.ndarray],
    spatial_engine: Optional[SpatialHandEngine] = None
) -> np.ndarray:
    """Computes the 151-D feature vector from raw left and right landmarks."""
    engine = spatial_engine or SpatialHandEngine()

    has_left = left_lm is not None and len(left_lm) == 21 and np.any(left_lm)
    has_right = right_lm is not None and len(right_lm) == 21 and np.any(right_lm)

    # 1. Normalized landmarks (42x3 -> 126)
    l_norm = left_lm if has_left else np.zeros((21, 3), dtype=np.float32)
    r_norm = right_lm if has_right else np.zeros((21, 3), dtype=np.float32)

    # Normalize centered on wrist
    if has_left:
        l_norm = l_norm - l_norm[0]
    if has_right:
        r_norm = r_norm - r_norm[0]

    dual_landmarks = np.concatenate([l_norm, r_norm], axis=0)  # (42, 3)
    landmarks_flat = dual_landmarks.flatten()  # (126,)

    # 2. Touch distance matrix (5x5 -> 25)
    touch_matrix = np.zeros((5, 5), dtype=np.float32)
    tip_indices = [4, 8, 12, 16, 20]

    if has_left and has_right:
        for i, l_idx in enumerate(tip_indices):
            for j, r_idx in enumerate(tip_indices):
                dist = np.linalg.norm(left_lm[l_idx] - right_lm[r_idx])
                touch_matrix[i, j] = np.exp(-dist * 5.0)

    touch_flat = touch_matrix.flatten()  # (25,)

    # Combine -> 151
    spatial_vector = np.concatenate([landmarks_flat, touch_flat]).astype(np.float32)
    return spatial_vector


def process_raw_dataset(
    raw_dir: Path = RAW_SAMPLES_DIR,
    out_dir: Path = OUTPUT_DIR
) -> int:
    """Extracts 151-D vectors for all classes in raw_dir and saves as .npy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = SpatialHandEngine()
    total_processed = 0

    logger.info(f"Extracting 151-D spatial vectors from {raw_dir} -> {out_dir}...")

    for sign_folder in raw_dir.glob("*"):
        if not sign_folder.is_dir():
            continue

        slug = sign_folder.name
        target_sign_dir = out_dir / slug
        target_sign_dir.mkdir(parents=True, exist_ok=True)

        for npy_file in sign_folder.glob("*.npy"):
            try:
                data = np.load(npy_file, allow_pickle=True).item()
                left_lm = np.array(data["left_landmarks"], dtype=np.float32) if data.get("left_landmarks") else None
                right_lm = np.array(data["right_landmarks"], dtype=np.float32) if data.get("right_landmarks") else None

                spatial_vec = extract_151d_from_landmarks(left_lm, right_lm, engine)
                out_path = target_sign_dir / npy_file.name
                np.save(out_path, spatial_vec)
                total_processed += 1
            except Exception as e:
                logger.debug(f"Error processing {npy_file}: {e}")

    # Consolidate into single .npz for instantaneous model training
    all_x = []
    all_y = []
    class_map = {}

    for c_idx, sign_folder in enumerate(sorted([d for d in out_dir.glob("*") if d.is_dir()])):
        slug = sign_folder.name
        class_map[c_idx] = slug
        for f in sign_folder.glob("*.npy"):
            try:
                vec = np.load(f)
                if vec.shape == (151,):
                    all_x.append(vec)
                    all_y.append(c_idx)
            except Exception:
                pass

    if all_x:
        npz_path = Path("dataset/spatial_dataset.npz")
        np.savez_compressed(npz_path, X=np.array(all_x, dtype=np.float32), y=np.array(all_y, dtype=np.int64))
        logger.info(f"Saved consolidated spatial dataset to {npz_path} ({len(all_x)} samples).")

    logger.info(f"151-D Extraction complete. Processed {total_processed} spatial samples.")
    return total_processed


if __name__ == "__main__":
    process_raw_dataset()
