"""External Image & Video Batch Ingestion & Tensor Processing Tool for IsharaConnect.

Batch processes arbitrary directories of raw BdSL images or MP4/AVI/MKV video files,
extracts 151-D spatial vectors or (60, 151) temporal tensors with multi-processing,
and exports compressed, validated .npz datasets.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core_engine.vision.spatial_hand_engine import SpatialHandEngine
from dataset.ingestors.tier1_fingerspelling_ingestor import Tier1FingerspellingIngestor
from dataset.ingestors.tier2_islr_ingestor import Tier2ISLRIngestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("batch_processor")


class BatchDatasetProcessor:
    """Multi-threaded Batch Ingestion Engine for External Image/Video Repositories."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self.spatial_engine = SpatialHandEngine()
        self.tier1_ingestor = Tier1FingerspellingIngestor()
        self.tier2_ingestor = Tier2ISLRIngestor()

    def process_image_file(self, img_path: Path) -> Optional[np.ndarray]:
        """Reads an image and extracts a normalized 151-D spatial landmark vector."""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            with self._lock:
                return self.tier1_ingestor.extract_from_image(img, augment=False)
        except Exception as e:
            logger.debug(f"Error processing image {img_path}: {e}")
            return None

    def process_video_file(self, video_path: Path, target_frames: int = 60) -> Optional[np.ndarray]:
        """Reads video frames, extracts landmarks per frame, and resamples to (60, 151)."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                return None

            with self._lock:
                return self.tier2_ingestor.extract_from_frames(frames, target_length=target_frames)
        except Exception as e:
            logger.debug(f"Error processing video {video_path}: {e}")
            return None

    def process_directory(
        self,
        input_dir: str,
        modality: str = "image",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Scans input directory for class subfolders or files and compiles compressed .npz dataset."""
        in_path = Path(input_dir)
        if not in_path.exists():
            return {"status": "error", "message": f"Input directory '{input_dir}' not found."}

        modality = modality.lower().strip()
        is_video = modality in ["video", "vid", "islr"]
        extensions = [".mp4", ".avi", ".mkv", ".mov"] if is_video else [".jpg", ".jpeg", ".png", ".webp"]

        # Discover class directories or flat files
        subdirs = [d for d in in_path.iterdir() if d.is_dir()]
        all_samples = []
        all_labels = []
        classes_map = {}

        if subdirs:
            # Class-folder hierarchy: input_dir/class_name/file.ext
            for class_idx, class_dir in enumerate(sorted(subdirs)):
                class_name = class_dir.name
                classes_map[class_idx] = class_name
                files = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]
                for f in files:
                    all_samples.append((f, class_idx))
        else:
            # Flat directory: input_dir/file.ext
            files = [f for f in in_path.iterdir() if f.suffix.lower() in extensions]
            for f in files:
                all_samples.append((f, 0))
            classes_map[0] = in_path.name

        if not all_samples:
            return {"status": "warning", "message": f"No matching {modality} files found in '{input_dir}'."}

        logger.info(f"Discovered {len(all_samples)} {modality} files across {len(classes_map)} classes.")

        extracted_X = []
        extracted_y = []

        # Parallel Worker Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if is_video:
                futures = {executor.submit(self.process_video_file, item[0]): item[1] for item in all_samples}
            else:
                futures = {executor.submit(self.process_image_file, item[0]): item[1] for item in all_samples}

            for future in concurrent.futures.as_completed(futures):
                class_label = futures[future]
                try:
                    tensor = future.result()
                    if tensor is not None:
                        extracted_X.append(tensor)
                        extracted_y.append(class_label)
                except Exception as e:
                    logger.debug(f"Processing error in worker: {e}")

        if not extracted_X:
            return {"status": "error", "message": "Failed to extract landmarks from any discovered files."}

        X = np.array(extracted_X, dtype=np.float32)
        y = np.array(extracted_y, dtype=np.int64)

        if output_file is None:
            out_name = f"batch_{modality}_dataset.npz"
            output_path = Path("dataset/processed") / out_name
        else:
            output_path = Path(output_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            classes_json=json.dumps(classes_map),
            modality=modality
        )

        logger.info(f"Successfully processed {len(X)} samples into {output_path} (Shape: {X.shape}).")
        return {
            "status": "success",
            "modality": modality,
            "samples_processed": int(len(X)),
            "classes_count": int(len(classes_map)),
            "tensor_shape": list(X.shape),
            "output_file": str(output_path)
        }


def main():
    parser = argparse.ArgumentParser(description="IsharaConnect External Dataset Batch Processor")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw image/video folder")
    parser.add_argument("--modality", type=str, choices=["image", "video"], default="image", help="Processing modality")
    parser.add_argument("--output_file", type=str, default=None, help="Destination .npz dataset path")
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel worker thread count")

    args = parser.parse_args()
    processor = BatchDatasetProcessor(max_workers=args.max_workers)
    result = processor.process_directory(args.input_dir, modality=args.modality, output_file=args.output_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
