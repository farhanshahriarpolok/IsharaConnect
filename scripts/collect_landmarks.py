"""Landmark Data Collection CLI Tool for Bangla Sign Language (BdSL).

Captures video stream from webcam, processes frames via MediaPipe Hands,
applies wrist coordinate normalization, and records structured sequences (.npy/JSON).
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

# Adjust sys.path so we can import from core_engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_engine.vision.hand_detector import HandDetector
from core_engine.preprocessing.normalizer import LandmarkNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("collect_landmarks")


def load_labels(labels_path: Path) -> dict:
    """Load BdSL labels dictionary."""
    if not labels_path.exists():
        logger.error("Labels file not found at: %s", labels_path)
        sys.exit(1)
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_ui(
    frame: np.ndarray,
    target_sign: dict,
    state: str,
    frames_recorded: int,
    total_frames: int,
    samples_recorded: int,
    total_samples: int,
    fps: int,
) -> None:
    """Draw interactive overlay on the video frame."""
    h, w, _ = frame.shape
    
    # Overlay background for text
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    
    # Sign Info
    sign_text = f"Sign: {target_sign['slug']} ({target_sign['category']})"
    cv2.putText(frame, sign_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Status
    color = (0, 255, 255) # Yellow for ready
    if "COUNTDOWN" in state:
        color = (0, 165, 255) # Orange
    elif "RECORDING" in state:
        color = (0, 0, 255) # Red
    elif "SAVED" in state:
        color = (0, 255, 0) # Green

    status_text = f"STATE: {state}"
    cv2.putText(frame, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Progress
    progress_text = f"Sample: {samples_recorded}/{total_samples} | Frame: {frames_recorded}/{total_frames}"
    cv2.putText(frame, progress_text, (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps}", (w - 120, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Instructions
    instructions = "Press SPACE to start recording. Press Q to quit."
    cv2.putText(frame, instructions, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect BdSL Hand Landmark Sequences")
    parser.add_argument("--label", type=str, help="Slug of sign to record (e.g. 'dhonnobad')")
    parser.add_argument("--user-id", type=str, default="user_01", help="Identifier for the user recording")
    parser.add_argument("--samples", type=int, default=30, help="Number of sequence samples to record")
    parser.add_argument("--frames-per-sample", type=int, default=30, help="Number of frames per dynamic sample")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument("--output-dir", type=str, default="dataset/raw_landmarks", help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Interactive sign selection mode")

    args = parser.parse_args()
    labels_file = Path("dataset/labels.json")
    labels_data = load_labels(labels_file)

    logger.info("Loaded %d BdSL signs from %s", labels_data.get("total_signs", 0), labels_file)

    if args.interactive:
        print("\nAvailable Bangla Sign Language Vocabulary:")
        for sign in labels_data.get("signs", []):
            print(f"  [{sign['id']:02d}] {sign['slug']:<20} | {sign['bangla']} ({sign['english']}) [{sign['category']}]")
        print("\nSelect a sign slug to record using: python scripts/collect_landmarks.py --label <slug>")
        return

    if not args.label:
        logger.error("Please provide --label <slug> or run with --interactive")
        sys.exit(1)

    target_sign = next((s for s in labels_data.get("signs", []) if s["slug"] == args.label), None)
    if not target_sign:
        logger.error("Sign '%s' not found in dataset/labels.json", args.label)
        sys.exit(1)

    logger.info("Target Sign: %s (%s - %s) [%s]", target_sign['bangla'], target_sign['slug'], target_sign['english'], target_sign['category'])
    
    output_dir = Path(args.output_dir) / str(target_sign['id'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving to %s", output_dir)

    # Initialize VideoCapture and Detector
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        logger.error("Could not open camera %d", args.camera_id)
        sys.exit(1)

    detector = HandDetector(max_num_hands=2)

    samples_recorded = 0
    frames_recorded = 0
    state = "READY"
    sequence_data: List[np.ndarray] = []
    
    countdown_start = 0.0

    # FPS calculation
    prev_frame_time = 0.0
    new_frame_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break
                
            frame = cv2.flip(frame, 1)  # Mirror view
            
            # FPS
            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time

            # Process vision
            annotated_frame = detector.find_hands(frame, draw=True)
            extraction = detector.extract_landmarks(frame.shape)
            
            # Generate feature vector
            feature_vector = LandmarkNormalizer.process_frame(
                extraction["raw_left"], extraction["raw_right"]
            )

            # State Machine
            if state == "READY":
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    state = "COUNTDOWN"
                    countdown_start = time.time()
                elif key == ord('q'):
                    break
                    
            elif state == "COUNTDOWN":
                elapsed = time.time() - countdown_start
                if elapsed < 1:
                    state = "COUNTDOWN (2)"
                elif elapsed < 2:
                    state = "COUNTDOWN (1)"
                else:
                    state = "RECORDING"
                    sequence_data = []
                    frames_recorded = 0
                cv2.waitKey(1)
                
            elif state.startswith("COUNTDOWN ("):
                elapsed = time.time() - countdown_start
                if elapsed >= 2:
                    state = "RECORDING"
                    sequence_data = []
                    frames_recorded = 0
                elif elapsed >= 1 and state == "COUNTDOWN (2)":
                    state = "COUNTDOWN (1)"
                cv2.waitKey(1)

            elif state == "RECORDING":
                sequence_data.append(feature_vector)
                frames_recorded += 1
                
                if frames_recorded >= args.frames_per_sample:
                    # Save sequence
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{args.user_id}_{timestamp}.npy"
                    filepath = output_dir / filename
                    np.save(filepath, np.array(sequence_data))
                    
                    # Save metadata
                    meta_path = output_dir / f"{args.user_id}_{timestamp}.json"
                    metadata = {
                        "sign_id": target_sign['id'],
                        "sign_slug": target_sign['slug'],
                        "user_id": args.user_id,
                        "timestamp": timestamp,
                        "frames": args.frames_per_sample,
                        "resolution": [frame.shape[1], frame.shape[0]],
                        "fps": fps
                    }
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)
                        
                    samples_recorded += 1
                    state = "SAVED"
                    countdown_start = time.time()
                cv2.waitKey(1)
                
            elif state == "SAVED":
                if time.time() - countdown_start > 1.0:
                    if samples_recorded >= args.samples:
                        logger.info("Finished recording %d samples.", args.samples)
                        break
                    state = "READY"
                cv2.waitKey(1)

            draw_ui(annotated_frame, target_sign, state, frames_recorded, args.frames_per_sample, samples_recorded, args.samples, fps)
            cv2.imshow("IsharaConnect - Landmark Collection Studio", annotated_frame)

            # Failsafe quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
