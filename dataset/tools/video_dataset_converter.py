"""Video-to-Landmark Ingestion Pipeline for IsharaConnect.

Slices raw BdSL MP4/MKV video files into normalized 30-frame 151D
spatial-temporal numpy sequences using SpatialHandEngine.
"""

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core_engine.vision.spatial_hand_engine import SpatialHandEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def resample_temporal_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    """Resamples a temporal sequence (T, D) to target_length using linear interpolation.

    Args:
        sequence: Array of shape (T, D).
        target_length: Desired number of frames in the output sequence.

    Returns:
        Resampled array of shape (target_length, D).
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or len(seq) == 0:
        feature_dim = seq.shape[1] if seq.ndim == 2 else 151
        return np.zeros((target_length, feature_dim), dtype=np.float32)

    curr_len, num_features = seq.shape
    if curr_len == target_length:
        return seq

    if curr_len == 1:
        return np.tile(seq, (target_length, 1))

    orig_indices = np.linspace(0.0, 1.0, num=curr_len, dtype=np.float32)
    target_indices = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)

    resampled = np.zeros((target_length, num_features), dtype=np.float32)
    for d in range(num_features):
        resampled[:, d] = np.interp(target_indices, orig_indices, seq[:, d])

    return resampled


def extract_151d_vector(features: Dict[str, Any]) -> np.ndarray:
    """Extracts a 151-D spatial vector from SpatialHandEngine output dictionary."""
    normalized_landmarks = features.get("normalized_landmarks", np.zeros((42, 3), dtype=np.float32))
    touch_matrix = features.get("touch_matrix", np.zeros((5, 5), dtype=np.float32))

    # Replace inf distances with bounded float (e.g. 10.0)
    touch_clean = np.nan_to_num(touch_matrix, nan=10.0, posinf=10.0, neginf=0.0).astype(np.float32)

    landmarks_flat = normalized_landmarks.flatten().astype(np.float32)
    touch_flat = touch_clean.flatten().astype(np.float32)

    return np.concatenate([landmarks_flat, touch_flat])


def process_video(
    video_path: str,
    engine: Optional[SpatialHandEngine] = None,
    target_frames: int = 30
) -> Optional[np.ndarray]:
    """Processes a single video file into a (target_frames, 151) feature matrix.

    Args:
        video_path: Absolute or relative path to video file.
        engine: Optional SpatialHandEngine instance.
        target_frames: Target sequence length (default 30).

    Returns:
        np.ndarray of shape (target_frames, 151) or None if video could not be read.
    """
    if engine is None:
        engine = SpatialHandEngine(static_image_mode=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open video file: %s", video_path)
        return None

    raw_sequence: List[np.ndarray] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            features = engine.extract_spatial_features(frame)
            vector_151d = extract_151d_vector(features)
            raw_sequence.append(vector_151d)
    finally:
        cap.release()

    if not raw_sequence:
        logger.warning("No frames extracted from video: %s", video_path)
        return None

    raw_array = np.array(raw_sequence, dtype=np.float32)
    return resample_temporal_sequence(raw_array, target_length=target_frames)


def convert_video_dataset(
    video_dir: str,
    output_dir: str,
    target_frames: int = 30,
    engine: Optional[SpatialHandEngine] = None
) -> Dict[str, Any]:
    """Scans video directory, extracts 151D dynamic sequences, and saves to output directory.

    Supports both directory formats:
      1. video_dir/<sign_label>/<video_file>.mp4
      2. video_dir/<sign_label_video>.mp4

    Args:
        video_dir: Input directory containing raw video files.
        output_dir: Output directory to save processed .npy files.
        target_frames: Target frame count per sequence (default 30).
        engine: Optional SpatialHandEngine instance.

    Returns:
        Summary dict containing counts and paths.
    """
    if engine is None:
        engine = SpatialHandEngine(static_image_mode=False)

    os.makedirs(output_dir, exist_ok=True)
    video_path_obj = Path(video_dir)

    if not video_path_obj.exists():
        logger.warning("Video directory %s does not exist.", video_dir)
        return {"processed": 0, "skipped": 0, "output_files": []}

    video_files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        video_files.extend(video_path_obj.rglob(f"*{ext}"))

    processed_count = 0
    skipped_count = 0
    output_files: List[str] = []

    for vpath in video_files:
        rel_path = vpath.relative_to(video_path_obj)
        # Class directory is parent if nested, or stem if at root
        if len(rel_path.parts) > 1:
            sign_class = rel_path.parts[0]
            sample_id = vpath.stem
        else:
            sign_class = vpath.stem.split("_")[0]
            sample_id = vpath.stem

        out_sign_dir = os.path.join(output_dir, sign_class)
        os.makedirs(out_sign_dir, exist_ok=True)
        out_npy_path = os.path.join(out_sign_dir, f"{sample_id}.npy")

        try:
            seq = process_video(str(vpath), engine=engine, target_frames=target_frames)
            if seq is not None and seq.shape == (target_frames, 151):
                np.save(out_npy_path, seq)
                output_files.append(out_npy_path)
                processed_count += 1
                logger.info("Processed: %s -> %s [shape: %s]", vpath.name, out_npy_path, seq.shape)
            else:
                skipped_count += 1
        except Exception as e:
            logger.error("Error processing %s: %s", vpath, e)
            skipped_count += 1

    summary = {
        "processed": processed_count,
        "skipped": skipped_count,
        "output_files": output_files,
        "target_frames": target_frames,
        "feature_dim": 151
    }
    logger.info(
        "Conversion complete: %d processed, %d skipped across %s",
        processed_count, skipped_count, video_dir
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="IsharaConnect Video to 151D Landmark Sequence Ingestor")
    parser.add_argument("--video_dir", type=str, default="dataset/raw_videos", help="Input directory of video files")
    parser.add_argument("--output_dir", type=str, default="dataset/dynamic_landmarks", help="Output directory for .npy")
    parser.add_argument("--target_frames", type=int, default=30, help="Temporal sequence length (default 30)")

    args = parser.parse_args()
    convert_video_dataset(args.video_dir, args.output_dir, target_frames=args.target_frames)


if __name__ == "__main__":
    main()
