"""Automated Retraining Worker for IsharaConnect.

Merges approved landmark sequences into dataset, updates dictionary,
and triggers PyTorch re-training and ONNX export.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("retrain_worker")


def update_dictionary(label_bn: str, label_en: str, contributor: str) -> int:
    """Append new entry to dataset/labels.json and return its new ID."""
    labels_file = Path("dataset/labels.json")
    
    with open(labels_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    signs = data.get("signs", [])
    
    # Check if already exists
    for sign in signs:
        if sign["bangla"] == label_bn:
            return sign["id"]
            
    # Add new
    new_id = len(signs)
    slug = label_en.lower().replace(" ", "_")
    
    new_sign = {
        "id": new_id,
        "bangla": label_bn,
        "english": label_en,
        "slug": slug,
        "category": "community_contributed",
        "contributor": contributor
    }
    
    signs.append(new_sign)
    data["signs"] = signs
    data["total_signs"] = len(signs)
    
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    logger.info("Added new sign '%s' to dictionary with ID %d", label_bn, new_id)
    return new_id


def process_approved_submission(submission_id: str) -> bool:
    """Process an approved submission and trigger retraining."""
    logger.info("Processing approved submission: %s", submission_id)
    
    submission_file = Path(f"dataset/pending_submissions/{submission_id}.json")
    if not submission_file.exists():
        logger.error("Submission file not found: %s", submission_file)
        return False
        
    with open(submission_file, "r", encoding="utf-8") as f:
        submission = json.load(f)
        
    if submission.get("status") != "APPROVED":
        logger.error("Submission %s is not marked APPROVED.", submission_id)
        return False
        
    label_bn = submission.get("label_bn")
    label_en = submission.get("label_en")
    contributor = submission.get("contributor")
    samples = submission.get("samples", [])
    
    # 1. Update Dictionary
    sign_id = update_dictionary(label_bn, label_en, contributor)
    
    # 2. Save Samples to raw dataset
    # In reality we save them to raw_landmarks and re-run normalization pipeline.
    # Here, samples are already extracted feature vectors of shape (30, 128) or similar.
    # We will save them as npy files.
    raw_dir = Path(f"dataset/raw_landmarks/{sign_id}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples):
        # sample is a list of lists, convert to numpy
        arr = np.array(sample, dtype=np.float32)
        file_path = raw_dir / f"{submission_id}_sample_{i}.npy"
        np.save(file_path, arr)
        
    logger.info("Saved %d samples to %s", len(samples), raw_dir)
    
    # 3. Trigger Retraining Pipeline
    logger.info("Triggering PyTorch retraining pipeline...")
    
    train_script = Path("scripts/train.py").resolve()
    try:
        subprocess.run([sys.executable, str(train_script)], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Training script failed: %s", e)
        return False
        
    # 4. Trigger ONNX Export
    logger.info("Triggering ONNX export...")
    
    export_script = Path("scripts/export_onnx.py").resolve()
    model_path = Path("models/checkpoints/bdsl_model_best.pth").resolve()
    
    # Need to get new total classes count
    with open(Path("dataset/labels.json"), "r", encoding="utf-8") as f:
        total_classes = json.load(f).get("total_signs", 24)
        
    try:
        subprocess.run([
            sys.executable, str(export_script), 
            "--model-path", str(model_path),
            "--num-classes", str(total_classes)
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("ONNX export script failed: %s", e)
        return False
        
    logger.info("Retraining pipeline completed successfully for %s.", submission_id)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, required=True, help="Submission ID to process")
    args = parser.parse_args()
    
    success = process_approved_submission(args.submission)
    if not success:
        sys.exit(1)
