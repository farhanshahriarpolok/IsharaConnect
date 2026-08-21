import os
import cv2
import json
import uuid
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core_engine.vision.hand_detector import HandDetector
from core_engine.preprocessing.normalizer import LandmarkNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MultiDatasetIngestor")


class MultiDatasetIngestor:
    def __init__(self, output_dir: str = "dataset/raw_landmarks", labels_path: str = "dataset/labels.json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path = Path(labels_path)
        
        self.labels_data = self._load_labels()
        self.slug_to_id = {s["slug"]: s["id"] for s in self.labels_data.get("signs", [])}
        
        self.detector = HandDetector(static_image_mode=True, max_num_hands=2)

    def _load_labels(self) -> dict:
        if self.labels_path.exists():
            with open(self.labels_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"signs": [], "total_signs": 0}

    def _get_or_create_class_id(self, slug: str) -> int:
        slug = slug.lower().strip()
        if slug in self.slug_to_id:
            return self.slug_to_id[slug]
            
        logger.info(f"Adding new class: {slug}")
        next_id = max(self.slug_to_id.values()) + 1 if self.slug_to_id else 0
        self.slug_to_id[slug] = next_id
        
        self.labels_data["signs"].append({
            "id": next_id,
            "slug": slug,
            "label_bn": f"অজ্ঞাত ({slug})",
            "label_en": slug.capitalize(),
            "tier": "external",
            "handedness": "unknown",
            "motion_type": "unknown"
        })
        self.labels_data["total_signs"] = len(self.labels_data["signs"])
        
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(self.labels_data, f, indent=2, ensure_ascii=False)
            
        return next_id

    def augment_static_image(self, image: np.ndarray, num_frames: int = 30) -> List[np.ndarray]:
        """Generates temporal sequence from static image with slight micro-variations."""
        sequence = []
        h, w = image.shape[:2]
        for _ in range(num_frames):
            angle = np.random.uniform(-3, 3)
            scale = np.random.uniform(0.98, 1.02)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            aug_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Add subtle jitter/noise
            noise = np.random.normal(0, 1.5, aug_img.shape).astype(np.uint8)
            aug_img = cv2.add(aug_img, noise)
            sequence.append(aug_img)
        return sequence

    def extract_video_frames(self, video_path: Path, target_frames: int = 30) -> List[np.ndarray]:
        """Extracts and samples uniform frames from a video clip."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            return []
            
        # Uniform sampling
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        return [frames[i] for i in indices]

    def process_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """Runs batch landmark extraction and normalization on a sequence of frames."""
        features = []
        for frame in frames:
            self.detector.find_hands(frame, draw=False)
            ext = self.detector.extract_landmarks(frame.shape)
            fv = LandmarkNormalizer.process_frame(ext["raw_left"], ext["raw_right"])
            features.append(fv)
        return np.array(features, dtype=np.float32)

    def ingest_directory(self, root_dir: str):
        """Auto-detects format and ingests Kaggle/INDORE image folders and video folders."""
        root_path = Path(root_dir)
        if not root_path.exists():
            logger.error(f"Directory {root_dir} does not exist.")
            return

        for class_dir in root_path.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
                
            class_slug = class_dir.name
            class_id = self._get_or_create_class_id(class_slug)
            out_class_dir = self.output_dir / str(class_id)
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Process static images (Kaggle / INDORE)
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            if image_files:
                logger.info(f"Processing {len(image_files)} static images for class {class_slug}")
                for img_path in image_files:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    frames = self.augment_static_image(img)
                    feature_seq = self.process_sequence(frames)
                    
                    sample_hash = uuid.uuid4().hex[:8]
                    np.save(out_class_dir / f"static_{sample_hash}.npy", feature_seq)
            
            # Process video clips
            video_files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
            if video_files:
                logger.info(f"Processing {len(video_files)} videos for class {class_slug}")
                for vid_path in video_files:
                    frames = self.extract_video_frames(vid_path)
                    if not frames:
                        continue
                    feature_seq = self.process_sequence(frames)
                    
                    sample_hash = uuid.uuid4().hex[:8]
                    np.save(out_class_dir / f"video_{sample_hash}.npy", feature_seq)

        logger.info("Ingestion completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset BdSL Ingestor")
    parser.add_argument("--source", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--output", type=str, default="dataset/raw_landmarks", help="Output directory for .npy sequences")
    args = parser.parse_args()
    
    ingestor = MultiDatasetIngestor(output_dir=args.output)
    ingestor.ingest_directory(args.source)

if __name__ == "__main__":
    main()
