"""Advanced Device & Camera Settings Dialog for IsharaConnect Desktop Client.

Provides live camera device probing, video resolution configuration,
inference confidence threshold adjustment, stability timer calibration,
TTS audio controls, and backend WebSocket server URL persistence.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from desktop_app.ui.theme import ThemeColors, ThemeStyles

logger = logging.getLogger(__name__)

SETTINGS_FILE_PATH = Path("config/user_settings.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "camera_id": 0,
    "resolution": [640, 480],
    "confidence_threshold": 0.70,
    "stability_timer": 2.0,
    "sensitivity": "normal",
    "mirror_mode": True,
    "tts_volume": 100,
    "tts_muted": False,
    "server_url": "ws://127.0.0.1:8000",
}


def load_user_settings() -> Dict[str, Any]:
    """Loads user configuration from config/user_settings.json with fallback to defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE_PATH.exists():
        try:
            with open(SETTINGS_FILE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                settings.update(saved)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", SETTINGS_FILE_PATH, e)
    return settings


def save_user_settings(settings: Dict[str, Any]) -> bool:
    """Persists user configuration to config/user_settings.json."""
    try:
        SETTINGS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        logger.info("Saved user settings to %s", SETTINGS_FILE_PATH)
        return True
    except Exception as e:
        logger.error("Failed to write %s: %s", SETTINGS_FILE_PATH, e)
        return False


class CameraScannerThread(QThread):
    """Probes video capture devices non-blockingly."""

    scan_finished = pyqtSignal(list)

    def run(self):
        found_devices: List[Tuple[int, str]] = []
        backends = [
            ("DirectShow", cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else ("Default", cv2.CAP_ANY),
            ("Default", cv2.CAP_ANY),
        ]

        for idx in range(3):
            for backend_name, backend_id in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend_id)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            found_devices.append((idx, f"Camera {idx} ({backend_name})"))
                            break
                except Exception:
                    pass

        if not found_devices:
            found_devices.append((0, "Camera 0 (Default Fallback)"))

        self.scan_finished.emit(found_devices)


class SettingsDialog(QDialog):
    """Advanced Device & System Configuration Modal."""

    settings_applied = pyqtSignal(dict)

    def __init__(self, current_settings: Optional[Dict[str, Any]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.settings = current_settings or load_user_settings()
        self.scanner_thread: Optional[CameraScannerThread] = None

        self._init_ui()
        self._load_values_into_ui()

    def _init_ui(self):
        self.setWindowTitle("Advanced Device & Inference Settings")
        self.setMinimumSize(580, 620)
        self.setStyleSheet("""
            QDialog {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 12px;
            }
            QLabel {
                color: #F8FAFC;
                font-family: 'Segoe UI', Arial;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 14px;
                font-size: 13px;
                font-weight: bold;
                color: #06B6D4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                left: 10px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #1E293B;
                color: #F8FAFC;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #06B6D4;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1E293B;
                color: #F8FAFC;
                selection-background-color: #06B6D4;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #334155;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #06B6D4;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #F8FAFC;
                border: 2px solid #0891B2;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
            QCheckBox {
                color: #CBD5E1;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator:checked {
                background-color: #10B981;
                border: 1px solid #059669;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #1E293B;
                color: #F8FAFC;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        # Header Title
        header = QLabel("⚙️ Device & Model Settings")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #06B6D4;")
        main_layout.addWidget(header)

        # Tab or Grouped Content
        # 1. Camera & Video Group
        cam_group = QGroupBox("📷 Camera & Vision Capture")
        cam_layout = QVBoxLayout(cam_group)
        cam_layout.setSpacing(10)

        # Camera selection row
        cam_row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.addItem("Camera 0 (Default)", 0)
        self.cam_combo.addItem("Camera 1 (External / Secondary)", 1)
        self.cam_combo.addItem("Camera 2 (USB / DirectShow)", 2)
        
        self.scan_btn = QPushButton("🔄 Scan Devices")
        self.scan_btn.clicked.connect(self._scan_cameras)
        cam_row.addWidget(QLabel("Capture Device:"))
        cam_row.addWidget(self.cam_combo, stretch=1)
        cam_row.addWidget(self.scan_btn)
        cam_layout.addLayout(cam_row)

        # Resolution and Mirror Mode
        res_row = QHBoxLayout()
        self.res_combo = QComboBox()
        self.res_combo.addItem("640 x 480 (Recommended for 30+ FPS)", [640, 480])
        self.res_combo.addItem("1280 x 720 (HD 720p)", [1280, 720])
        self.res_combo.addItem("1920 x 1080 (Full HD 1080p)", [1920, 1080])

        self.cb_mirror = QCheckBox("Horizontal Flip (Mirror Mode)")
        self.cb_mirror.setChecked(True)

        res_row.addWidget(QLabel("Video Resolution:"))
        res_row.addWidget(self.res_combo, stretch=1)
        cam_layout.addLayout(res_row)
        cam_layout.addWidget(self.cb_mirror)
        main_layout.addWidget(cam_group)

        # 2. AI Inference & Temporal Calibration Group
        ai_group = QGroupBox("🧠 AI Inference & Posture Calibration")
        ai_layout = QVBoxLayout(ai_group)
        ai_layout.setSpacing(12)

        # Confidence Threshold Slider (50 to 95 -> 0.50 to 0.95)
        conf_row = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(50, 95)
        self.conf_slider.setValue(70)
        self.conf_val_lbl = QLabel("0.70")
        self.conf_val_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.conf_val_lbl.setStyleSheet("color: #10B981; min-width: 45px;")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_val_lbl.setText(f"{v / 100.0:.2f}"))

        conf_row.addWidget(QLabel("Min Confidence Threshold:"))
        conf_row.addWidget(self.conf_slider, stretch=1)
        conf_row.addWidget(self.conf_val_lbl)
        ai_layout.addLayout(conf_row)

        # Posture Hold Stability Timer
        timer_row = QHBoxLayout()
        self.hold_timer_spin = QDoubleSpinBox()
        self.hold_timer_spin.setRange(0.5, 5.0)
        self.hold_timer_spin.setSingleStep(0.5)
        self.hold_timer_spin.setValue(2.0)
        self.hold_timer_spin.setSuffix(" sec")

        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItem("Normal (Balanced)", "normal")
        self.sensitivity_combo.addItem("High (Rapid Gestures)", "high")
        self.sensitivity_combo.addItem("Strict (Precision Training)", "strict")

        timer_row.addWidget(QLabel("Hold Stability Duration:"))
        timer_row.addWidget(self.hold_timer_spin)
        timer_row.addSpacing(15)
        timer_row.addWidget(QLabel("Preset Sensitivity:"))
        timer_row.addWidget(self.sensitivity_combo)
        ai_layout.addLayout(timer_row)
        main_layout.addWidget(ai_group)

        # 3. Audio & Network Group
        net_group = QGroupBox("🔊 Audio Synthesis & Backend Server")
        net_layout = QVBoxLayout(net_group)
        net_layout.setSpacing(10)

        # Volume control
        vol_row = QHBoxLayout()
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(100)
        self.vol_val_lbl = QLabel("100%")
        self.vol_val_lbl.setStyleSheet("color: #06B6D4; min-width: 40px;")
        self.vol_slider.valueChanged.connect(lambda v: self.vol_val_lbl.setText(f"{v}%"))

        self.cb_mute = QCheckBox("Mute Audio")
        vol_row.addWidget(QLabel("TTS Speech Volume:"))
        vol_row.addWidget(self.vol_slider, stretch=1)
        vol_row.addWidget(self.vol_val_lbl)
        vol_row.addWidget(self.cb_mute)
        net_layout.addLayout(vol_row)

        # Backend URL
        url_row = QHBoxLayout()
        self.server_url_input = QLineEdit("ws://127.0.0.1:8000")
        self.server_url_input.setPlaceholderText("ws://127.0.0.1:8000")
        url_row.addWidget(QLabel("WebSocket Server URL:"))
        url_row.addWidget(self.server_url_input, stretch=1)
        net_layout.addLayout(url_row)
        main_layout.addWidget(net_group)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #0284C7;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #0EA5E9;
            }
        """)
        self.apply_btn.clicked.connect(self._apply_settings)

        self.save_btn = QPushButton("💾 Save & Close")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: #064E3B;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #34D399;
            }
        """)
        self.save_btn.clicked.connect(self._save_and_close)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.save_btn)
        main_layout.addLayout(btn_layout)

    def _load_values_into_ui(self):
        # Camera
        cam_id = self.settings.get("camera_id", 0)
        found_idx = self.cam_combo.findData(cam_id)
        if found_idx >= 0:
            self.cam_combo.setCurrentIndex(found_idx)

        # Resolution
        res = self.settings.get("resolution", [640, 480])
        for i in range(self.res_combo.count()):
            if self.res_combo.itemData(i) == res:
                self.res_combo.setCurrentIndex(i)
                break

        # Confidence
        conf = int(self.settings.get("confidence_threshold", 0.70) * 100)
        self.conf_slider.setValue(max(50, min(95, conf)))

        # Hold timer
        self.hold_timer_spin.setValue(self.settings.get("stability_timer", 2.0))

        # Sensitivity
        sens = self.settings.get("sensitivity", "normal")
        sens_idx = self.sensitivity_combo.findData(sens)
        if sens_idx >= 0:
            self.sensitivity_combo.setCurrentIndex(sens_idx)

        # Mirror
        self.cb_mirror.setChecked(self.settings.get("mirror_mode", True))

        # TTS Volume & Mute
        vol = self.settings.get("tts_volume", 100)
        self.vol_slider.setValue(vol)
        self.cb_mute.setChecked(self.settings.get("tts_muted", False))

        # Server URL
        self.server_url_input.setText(self.settings.get("server_url", "ws://127.0.0.1:8000"))

    def _get_current_settings_from_ui(self) -> Dict[str, Any]:
        return {
            "camera_id": self.cam_combo.currentData() if self.cam_combo.currentData() is not None else 0,
            "resolution": self.res_combo.currentData() or [640, 480],
            "confidence_threshold": round(self.conf_slider.value() / 100.0, 2),
            "stability_timer": round(self.hold_timer_spin.value(), 1),
            "sensitivity": self.sensitivity_combo.currentData() or "normal",
            "mirror_mode": self.cb_mirror.isChecked(),
            "tts_volume": self.vol_slider.value(),
            "tts_muted": self.cb_mute.isChecked(),
            "server_url": self.server_url_input.text().strip() or "ws://127.0.0.1:8000",
        }

    def _scan_cameras(self):
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")
        self.scanner_thread = CameraScannerThread()
        self.scanner_thread.scan_finished.connect(self._on_scan_finished)
        self.scanner_thread.start()

    @pyqtSlot(list)
    def _on_scan_finished(self, devices: List[Tuple[int, str]]):
        self.cam_combo.clear()
        for idx, name in devices:
            self.cam_combo.addItem(name, idx)
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("🔄 Scan Devices")
        logger.info("Camera scan finished: found %d devices.", len(devices))

    def _apply_settings(self):
        new_settings = self._get_current_settings_from_ui()
        self.settings.update(new_settings)
        self.settings_applied.emit(new_settings)
        logger.info("Settings applied: %s", new_settings)

    def _save_and_close(self):
        self._apply_settings()
        save_user_settings(self.settings)
        self.accept()
