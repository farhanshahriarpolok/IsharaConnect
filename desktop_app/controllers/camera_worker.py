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

logger = logging.getLogger(__name__)


class CameraWorker(QThread):
    """Worker thread for reading camera frames, running MediaPipe, and predicting."""

    # Signals
    frame_ready = pyqtSignal(QImage)
    sign_detected = pyqtSignal(dict)
    fps_updated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self._is_running = True
        
        self.detector: Optional[HandDetector] = None
        self.predictor: Optional[RealTimePredictor] = None

    def run(self):
        """Main loop for camera processing."""
        # Initialize ML engines inside the thread to avoid context issues
        try:
            self.detector = HandDetector(max_num_hands=2)
            self.predictor = RealTimePredictor()
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize ML engines: {e}")
            return

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera {self.camera_id}")
            # Fallback to dummy frame generator
            self._run_dummy_loop()
            return

        prev_time = time.time()
        
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                self.error_occurred.emit("Failed to grab frame.")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)  # Mirror view
            
            # 1. Vision Processing
            annotated_frame = self.detector.find_hands(frame, draw=True)
            extraction = self.detector.extract_landmarks(frame.shape)
            
            # 2. Normalization
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

        cap.release()
        self.detector.close()

    def _run_dummy_loop(self):
        """Fallback loop if camera is unavailable."""
        logger.warning("Running dummy camera loop.")
        prev_time = time.time()
        while self._is_running:
            # Create a blank image with some text
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA UNAVAILABLE", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            self.fps_updated.emit(fps)
            
            q_img = QImage(frame.data, 640, 480, 3 * 640, QImage.Format.Format_RGB888)
            self.frame_ready.emit(q_img.copy())
            QThread.msleep(33) # roughly 30 fps

    def stop(self):
        """Stop the worker thread gracefully."""
        self._is_running = False
        self.wait()
