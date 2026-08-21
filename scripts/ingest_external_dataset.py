import os
import cv2
import json
import logging
import argparse
import subprocess
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_engine.vision.hand_detector import HandDetector
from core_engine.preprocessing.normalizer import LandmarkNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_dataset")

def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply spatial augmentation (rotation, scale, noise) to create a temporal variation."""
    h, w = image.shape[:2]
    
    # Random rotation (-5 to 5 degrees)
    angle = np.random.uniform(-5, 5)
    # Random scale (0.95 to 1.05)
    scale = np.random.uniform(0.95, 1.05)
    
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    aug_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Add slight Gaussian noise
    noise = np.random.normal(0, 2, aug_img.shape).astype(np.uint8)
    aug_img = cv2.add(aug_img, noise)
    
    return aug_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="https://github.com/Patchwork53/BdSL40_Dataset_AI_for_Bangla_2.0_Honorable_Mention.git")
    parser.add_argument("--dest", type=str, default="dataset/external/BdSL40")
    parser.add_argument("--out", type=str, default="dataset/raw_landmarks")
    args = parser.parse_args()

    dest_path = Path(args.dest)
    out_path = Path(args.out)
    
    if not dest_path.exists():
        logger.info(f"Cloning dataset from {args.repo} into {dest_path}...")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["C:\\Program Files\\Git\\cmd\\git.exe", "clone", args.repo, str(dest_path)], check=True)
    else:
        logger.info(f"Dataset already exists at {dest_path}")
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    labels_file = Path("dataset/labels.json")
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
    else:
        labels_data = {"signs": [], "total_signs": 0}
        
    slug_to_id = {s["slug"]: s["id"] for s in labels_data["signs"]}
    next_id = max(slug_to_id.values()) + 1 if slug_to_id else 0
    
    detector = HandDetector(static_image_mode=True, max_num_hands=2)
    
    # Scan dataset
    # We assume folder names represent the class label (slug)
    # E.g., BdSL40/A/image1.jpg
    valid_exts = {".jpg", ".jpeg", ".png"}
    
    for class_dir in dest_path.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
            
        slug = class_dir.name.lower()
        if slug not in slug_to_id:
            logger.info(f"Adding new class to vocabulary: {slug}")
            slug_to_id[slug] = next_id
            labels_data["signs"].append({
                "id": next_id,
                "slug": slug,
                "bangla": f"অজ্ঞাত ({slug})", # Placeholder for Bengali text
                "english": slug.capitalize(),
                "category": "External Dataset"
            })
            next_id += 1
            
        class_id = slug_to_id[slug]
        class_out_dir = out_path / str(class_id)
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        img_files = [f for f in class_dir.glob("**/*") if f.suffix.lower() in valid_exts]
        logger.info(f"Found {len(img_files)} images for class '{slug}'")
        
        for idx, img_path in enumerate(img_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            sequence = []
            valid = True
            
            # Generate 30 frames from single image using augmentation
            for _ in range(30):
                aug_img = augment_image(img)
                detector.find_hands(aug_img, draw=False)
                extraction = detector.extract_landmarks(aug_img.shape)
                
                # If no hands detected, we can't build a good sequence.
                # However, for robustness, we'll process anyway, normalizer sets to 0.
                fv = LandmarkNormalizer.process_frame(extraction["raw_left"], extraction["raw_right"])
                sequence.append(fv)
                
            sequence = np.array(sequence, dtype=np.float32)
            
            # Save .npy
            npy_path = class_out_dir / f"ext_{slug}_{idx}.npy"
            np.save(npy_path, sequence)
            
    detector.close()
    
    # Update labels.json
    labels_data["total_signs"] = len(labels_data["signs"])
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)
        
    logger.info("Dataset ingestion and landmark extraction complete!")

if __name__ == "__main__":
    main()
