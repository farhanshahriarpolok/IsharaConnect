"""Extractor script for 151D Spatial Dataset.

Iterates over dataset/external/ and dataset/raw_landmarks/, extracts 151-D
features using SpatialHandEngine, flattens them, and saves them as .npy arrays.
"""

import os
import glob
import logging
import cv2
import numpy as np
from pathlib import Path

from core_engine.vision.spatial_hand_engine import SpatialHandEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_dataset():
    """Extract 151-D features from datasets and save as .npy."""
    base_dirs = ["dataset/external", "dataset/raw_landmarks"]
    output_dir = "dataset/spatial_landmarks"
    
    engine = SpatialHandEngine(static_image_mode=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_processed = 0
    total_skipped = 0
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            logger.warning(f"Directory {base_dir} does not exist. Skipping.")
            continue
            
        # Assuming format is dataset_dir/sign_slug/image_file.jpg
        for sign_dir in os.listdir(base_dir):
            sign_path = os.path.join(base_dir, sign_dir)
            if not os.path.isdir(sign_path):
                continue
                
            out_sign_dir = os.path.join(output_dir, sign_dir)
            os.makedirs(out_sign_dir, exist_ok=True)
            
            image_files = []
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                image_files.extend(glob.glob(os.path.join(sign_path, ext)))
                
            for img_file in image_files:
                sample_id = Path(img_file).stem
                out_file = os.path.join(out_sign_dir, f"{sample_id}.npy")
                
                # If already exists, skip
                if os.path.exists(out_file):
                    continue
                    
                frame = cv2.imread(img_file)
                if frame is None:
                    logger.warning(f"Failed to read image {img_file}")
                    total_skipped += 1
                    continue
                    
                features = engine.extract_spatial_features(frame)
                
                if not (features["has_left"] or features["has_right"]):
                    logger.debug(f"No hands detected in {img_file}")
                    total_skipped += 1
                    continue
                
                # Flatten the 42x3 normalized landmarks -> 126
                normalized_landmarks_flat = features["normalized_landmarks"].flatten()
                
                # Flatten the 5x5 touch matrix -> 25
                touch_matrix_flat = features["touch_matrix"].flatten()
                
                # Concatenate -> 151
                spatial_vector = np.concatenate([normalized_landmarks_flat, touch_matrix_flat])
                
                np.save(out_file, spatial_vector)
                total_processed += 1
                
    logger.info(f"Extraction complete. Processed {total_processed} samples, skipped {total_skipped}.")

if __name__ == "__main__":
    extract_dataset()
