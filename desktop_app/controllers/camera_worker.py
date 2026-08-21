"""Camera worker thread for non-blocking capture and inference."""

import logging
import time
from typing import Optional, Dict

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from core_engine.vision.hand_detector import HandDetector
from core_engine.preprocessing.normalizer import LandmarkNormalizer
from core_engine.inference.predictor import RealTimePredictor
from core_engine.vision.spatial_hand_engine import SpatialHandEngine

logger = logging.getLogger(__name__)


class CameraWorker(QThread):
    """Worker thread for reading camera frames, running MediaPipe, and predicting."""

    # Signals
    frame_ready = pyqtSignal(QImage)
    sign_detected = pyqtSignal(dict)
    trajectory_ready = pyqtSignal(dict)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self._is_running = True
        
        self.detector: Optional[HandDetector] = None
        self.spatial_engine: Optional[SpatialHandEngine] = None
        self.predictor: Optional[RealTimePredictor] = None

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Probes multiple indices and backends to find an active camera."""
        indices = [self.camera_id, 0, 1, 2]
        backends = [
            ("DirectShow", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY)
        ]
        
        for idx in set(indices):  # Unique indices
            for backend_name, backend_id in backends:
                logger.info(f"Probing camera index {idx} with {backend_name}...")
                cap = cv2.VideoCapture(idx, backend_id)
                if cap.isOpened():
                    # Attempt to read a frame to confirm it's actually working
                    ret, _ = cap.read()
                    if ret:
                        logger.info(f"Successfully opened camera {idx} with {backend_name}.")
                        return cap
                    cap.release()
                
        return None

    def run(self):
        """Main loop for camera processing."""
        try:
            logger.info("Initializing MediaPipe and ONNX engines...")
            # Initialize ML engines inside the thread to avoid context issues
            self.detector = HandDetector(max_num_hands=2)
            self.spatial_engine = SpatialHandEngine()
            self.predictor = RealTimePredictor()
            logger.info("ML engines initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ML engines: {e}")
            self.error_occurred.emit(f"Failed to initialize ML engines: {e}")
            return

        try:
            logger.info("Attempting to open camera...")
            cap = self._open_camera()
            
            if not cap or not cap.isOpened():
                logger.error("No valid camera found across probed indices and backends.")
                self.error_occurred.emit(f"Cannot open camera (tried indices 0,1,2). Running synthetic feed.")
                # Fallback to dummy frame generator
                self._run_dummy_loop()
                return
                
            # Configure camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            prev_time = time.time()
            failed_frames = 0
            first_frame_read = False
            
            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    failed_frames += 1
                    if failed_frames > 30:
                        logger.error("Camera lost or locked by another process.")
                        self.error_occurred.emit("Camera lost or locked by another process.")
                        break
                    time.sleep(0.1)
                    continue
                    
                if not first_frame_read:
                    logger.info("First camera frame read successfully.")
                    first_frame_read = True

                failed_frames = 0
                frame = cv2.flip(frame, 1)  # Mirror view
                
                # 1. Vision Processing
                annotated_frame = self.detector.find_hands(frame, draw=True)
                extraction = self.detector.extract_landmarks(frame.shape)
                
                # 2. Extract Trajectory Points
                left_w, right_w, left_idx, right_idx = None, None, None, None
                if extraction["raw_left"] is not None and len(extraction["raw_left"]) >= 9:
                    left_w = (float(extraction["raw_left"][0][0]), float(extraction["raw_left"][0][1]))
                    left_idx = (float(extraction["raw_left"][8][0]), float(extraction["raw_left"][8][1]))
                if extraction["raw_right"] is not None and len(extraction["raw_right"]) >= 9:
                    right_w = (float(extraction["raw_right"][0][0]), float(extraction["raw_right"][0][1]))
                    right_idx = (float(extraction["raw_right"][8][0]), float(extraction["raw_right"][8][1]))
                
                self.trajectory_ready.emit({
                    "left_wrist": left_w,
                    "right_wrist": right_w,
                    "left_index": left_idx,
                    "right_index": right_idx
                })

                # Check for two hands
                if extraction["raw_left"] is not None and extraction["raw_right"] is not None:
                    # Dual-hand: get 151-D vector
                    spatial_features = self.spatial_engine.extract_spatial_features(frame)
                    normalized_landmarks_flat = spatial_features["normalized_landmarks"].flatten()
                    touch_matrix_flat = spatial_features["touch_matrix"].flatten()
                    feature_vector = np.concatenate([normalized_landmarks_flat, touch_matrix_flat])
                else:
                    # Normalization
                    feature_vector = LandmarkNormalizer.process_frame(
                        extraction["raw_left"], extraction["raw_right"]
                    )
                
                # 3. Inference
                prediction = self.predictor.process_frame(feature_vector)
                if prediction:
                    self.sign_detected.emit(prediction)

                # 4. FPS Calculation
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                self.fps_updated.emit(fps)

                # Convert frame to QImage
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                self.frame_ready.emit(q_img.copy())  # Must copy since memory is unmanaged after this scope
                
                # Yield to event loop
                QThread.msleep(10)

        except Exception as e:
            logger.exception(f"Unhandled exception in CameraWorker loop: {e}")
            self.error_occurred.emit(f"Camera error: {e}")
        finally:
            if 'cap' in locals() and cap and cap.isOpened():
                cap.release()
            if self.detector:
                self.detector.close()

    def _generate_fallback_frame(self, t: float) -> np.ndarray:
        import math
        from datetime import datetime
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw moving target
        cx = int(320 + math.sin(t) * 100)
        cy = int(240 + math.cos(t * 1.5) * 50)
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), 2)
        cv2.line(frame, (cx - 40, cy), (cx + 40, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 40), (cx, cy + 40), (0, 255, 0), 1)
        
        # Draw text
        cv2.putText(frame, "TEST FEED: No Camera Detected", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def _run_dummy_loop(self):
        """Fallback loop if camera is unavailable."""
        logger.warning("Running dummy camera loop.")
        prev_time = time.time()
        start_time = time.time()
        
        while self._is_running:
            t = time.time() - start_time
            frame = self._generate_fallback_frame(t)
            
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            self.fps_updated.emit(fps)
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            self.frame_ready.emit(q_img.copy())
            QThread.msleep(33) # roughly 30 fps

    def stop(self):
        """Stop the worker thread gracefully."""
        self._is_running = False
        self.wait()
