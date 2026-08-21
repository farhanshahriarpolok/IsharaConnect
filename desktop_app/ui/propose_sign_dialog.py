"""Dialog for proposing new signs."""

import logging
import json
import urllib.request
from typing import List
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QProgressBar, QMessageBox
)

from core_engine.vision.hand_detector import HandDetector
from core_engine.preprocessing.normalizer import LandmarkNormalizer
from desktop_app.utils.url_helpers import get_http_url

logger = logging.getLogger(__name__)


class ProposeSignDialog(QDialog):
    """Interactive dialog to capture and propose a new BdSL sign."""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        super().__init__()
        self.server_url = server_url
        self.samples_required = 5
        self.frames_per_sample = 30
        
        self.recorded_samples: List[List[List[float]]] = []
        self.current_sample_frames: List[List[float]] = []
        
        self.is_recording = False
        
        self._init_ui()
        
        try:
            self.detector = HandDetector(max_num_hands=2)
            self.cap = cv2.VideoCapture(0)
        except Exception as e:
            logger.error("Failed to init camera/detector: %s", e)
            self.cap = None
            
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        if self.cap and self.cap.isOpened():
            self.timer.start(33)

    def _init_ui(self):
        self.setWindowTitle("Propose New Sign")
        self.setMinimumSize(700, 500)
        self.setStyleSheet("""
            QDialog { background-color: #1E1E2E; color: #CDD6F4; }
            QLabel { color: #CDD6F4; font-weight: bold; }
            QLineEdit { background-color: #181825; color: #CDD6F4; border: 1px solid #313244; padding: 5px; }
            QPushButton { background-color: #89B4FA; color: #11111B; font-weight: bold; padding: 8px; border-radius: 4px; }
            QPushButton:disabled { background-color: #45475A; color: #6C7086; }
            QProgressBar { border: 1px solid #313244; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #A6E3A1; }
        """)
        
        layout = QVBoxLayout(self)
        
        # Form
        form_layout = QHBoxLayout()
        self.bn_input = QLineEdit()
        self.bn_input.setPlaceholderText("Bangla Sign Name (e.g. ধন্যবাদ)")
        self.en_input = QLineEdit()
        self.en_input.setPlaceholderText("English Sign Name (e.g. Thank You)")
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Your Name/ID")
        
        form_layout.addWidget(self.bn_input)
        form_layout.addWidget(self.en_input)
        form_layout.addWidget(self.user_input)
        layout.addLayout(form_layout)
        
        # Camera Feed
        self.camera_lbl = QLabel("Initializing Camera...")
        self.camera_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_lbl.setMinimumSize(640, 480)
        self.camera_lbl.setStyleSheet("background-color: black;")
        layout.addWidget(self.camera_lbl)
        
        # Progress & Controls
        controls = QHBoxLayout()
        self.record_btn = QPushButton(f"Record {self.samples_required} Samples")
        self.record_btn.clicked.connect(self._start_recording)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.samples_required * self.frames_per_sample)
        
        self.submit_btn = QPushButton("Submit to Admin")
        self.submit_btn.setEnabled(False)
        self.submit_btn.clicked.connect(self._submit_proposal)
        
        controls.addWidget(self.record_btn)
        controls.addWidget(self.progress_bar)
        controls.addWidget(self.submit_btn)
        layout.addLayout(controls)

    @pyqtSlot()
    def _update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        annotated_frame = self.detector.find_hands(frame, draw=True)
        
        if self.is_recording:
            extraction = self.detector.extract_landmarks(frame.shape)
            feature_vector = LandmarkNormalizer.process_frame(
                extraction["raw_left"], extraction["raw_right"]
            )
            
            # feature_vector is shape (128,)
            self.current_sample_frames.append(feature_vector.tolist())
            
            if len(self.current_sample_frames) >= self.frames_per_sample:
                self.recorded_samples.append(self.current_sample_frames)
                self.current_sample_frames = []
                
            total_frames = (len(self.recorded_samples) * self.frames_per_sample) + len(self.current_sample_frames)
            self.progress_bar.setValue(total_frames)
            
            if len(self.recorded_samples) >= self.samples_required:
                self.is_recording = False
                self.record_btn.setText("Recording Complete")
                self.record_btn.setEnabled(False)
                self.submit_btn.setEnabled(True)
                
        # Draw overlay if recording
        if self.is_recording:
            cv2.putText(annotated_frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.camera_lbl.setPixmap(pixmap)

    @pyqtSlot()
    def _start_recording(self):
        if not self.bn_input.text().strip() or not self.en_input.text().strip():
            QMessageBox.warning(self, "Missing Info", "Please provide Bangla and English labels.")
            return
            
        self.recorded_samples = []
        self.current_sample_frames = []
        self.progress_bar.setValue(0)
        self.is_recording = True
        self.record_btn.setText("Recording...")
        self.record_btn.setEnabled(False)

    @pyqtSlot()
    def _submit_proposal(self):
        payload = {
            "user_id": self.user_input.text().strip() or "anonymous",
            "bangla": self.bn_input.text().strip(),
            "english": self.en_input.text().strip(),
            "category": "proposed",
            "samples": self.recorded_samples
        }
        
        try:
            http_url = get_http_url(self.server_url, "/api/v1/signs/propose")
            req = urllib.request.Request(
                http_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    QMessageBox.information(self, "Success", "Sign proposed successfully! Awaiting admin review.")
                    self.accept()
                else:
                    QMessageBox.critical(self, "Error", f"Failed to submit: HTTP {response.status}")
        except Exception as e:
            logger.error("Submit failed: %s", e)
            QMessageBox.critical(self, "Error", f"Failed to submit: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        if hasattr(self, 'detector'):
            self.detector.close()
        event.accept()
