"""Modern PyQt6 MainWindow for IsharaConnect Desktop Client."""

import logging
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QLineEdit, QComboBox, 
    QFrame, QStackedWidget, QTextEdit, QProgressBar
)

from desktop_app.controllers.camera_worker import CameraWorker
from desktop_app.controllers.network_worker import NetworkWorker
from desktop_app.ui.propose_sign_dialog import ProposeSignDialog

logger = logging.getLogger(__name__)

# Modern Dark Palette
BG_COLOR = "#1E1E2E"
PANEL_COLOR = "#2A2B3D"
TEXT_COLOR = "#CDD6F4"
ACCENT_BLUE = "#89B4FA"
ACCENT_GREEN = "#A6E3A1"
ACCENT_RED = "#F38BA8"
BORDER_RADIUS = "8px"

STYLESHEET = f"""
    QMainWindow {{ background-color: {BG_COLOR}; }}
    QLabel {{ color: {TEXT_COLOR}; font-family: 'Segoe UI', Arial; }}
    QFrame {{ background-color: {PANEL_COLOR}; border-radius: {BORDER_RADIUS}; }}
    QPushButton {{ 
        background-color: {ACCENT_BLUE}; 
        color: #11111B; 
        font-weight: bold; 
        border-radius: {BORDER_RADIUS}; 
        padding: 8px 16px; 
    }}
    QPushButton:hover {{ background-color: #74C7EC; }}
    QPushButton:disabled {{ background-color: #45475A; color: #6C7086; }}
    QLineEdit, QComboBox, QTextEdit {{ 
        background-color: #181825; 
        color: {TEXT_COLOR}; 
        border: 1px solid #313244; 
        border-radius: 4px; 
        padding: 6px; 
    }}
    QProgressBar {{ 
        border: 1px solid #313244; 
        border-radius: 4px; 
        text-align: center; 
        color: white; 
    }}
    QProgressBar::chunk {{ background-color: {ACCENT_GREEN}; width: 10px; }}
"""


class IsharaMainWindow(QMainWindow):
    """Main Application Window."""

    def __init__(self, mode: str, room_id: str, server_url: str):
        super().__init__()
        self.mode = mode
        self.room_id = room_id
        self.server_url = server_url
        
        self.camera_worker = None
        self.network_worker = None
        
        self._init_ui()
        self._start_workers()

    def _init_ui(self):
        """Initialize the modern UI layout."""
        self.setWindowTitle(f"IsharaConnect (ইশারা কানেক্ট) - {self.mode.title()} Mode")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(STYLESHEET)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Header ---
        header_layout = QHBoxLayout()
        title_label = QLabel("IsharaConnect")
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {ACCENT_BLUE};")
        
        self.status_badge = QLabel("🔴 Offline")
        self.status_badge.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        
        self.room_input = QLineEdit(self.room_id)
        self.room_input.setPlaceholderText("Room ID")
        self.room_input.setMaximumWidth(150)
        
        self.join_btn = QPushButton("Reconnect")
        self.join_btn.clicked.connect(self._reconnect)
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Signer", "Speaker"])
        self.mode_selector.setCurrentText(self.mode.title())
        self.mode_selector.currentTextChanged.connect(self._change_mode)
        
        self.propose_btn = QPushButton("➕ Propose New Sign")
        self.propose_btn.clicked.connect(self._open_propose_dialog)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)
        header_layout.addWidget(self.room_input)
        header_layout.addWidget(self.join_btn)
        header_layout.addWidget(self.mode_selector)
        header_layout.addWidget(self.propose_btn)
        
        main_layout.addLayout(header_layout)

        # --- Dynamic View Area ---
        self.stacked_widget = QStackedWidget()
        self.signer_view = self._create_signer_view()
        self.speaker_view = self._create_speaker_view()
        
        self.stacked_widget.addWidget(self.signer_view)
        self.stacked_widget.addWidget(self.speaker_view)
        
        if self.mode.lower() == "signer":
            self.stacked_widget.setCurrentWidget(self.signer_view)
        else:
            self.stacked_widget.setCurrentWidget(self.speaker_view)
            
        main_layout.addWidget(self.stacked_widget)

    def _create_signer_view(self) -> QWidget:
        """Create the layout for Deaf/Mute users."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top: Camera + Local Sign Detection Status
        top_layout = QHBoxLayout()
        
        # Camera Feed
        self.camera_label = QLabel("Camera Feed Initializing...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet(f"background-color: #000000; border-radius: {BORDER_RADIUS};")
        self.camera_label.setMinimumSize(640, 480)
        
        # Status Panel
        status_panel = QFrame()
        status_panel.setFixedWidth(250)
        status_layout = QVBoxLayout(status_panel)
        
        status_layout.addWidget(QLabel("Current Sign (Local):"))
        self.local_sign_lbl = QLabel("---")
        self.local_sign_lbl.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.local_sign_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.local_sign_lbl.setStyleSheet(f"color: {ACCENT_GREEN};")
        status_layout.addWidget(self.local_sign_lbl)
        
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        status_layout.addWidget(self.conf_bar)
        
        status_layout.addStretch()
        self.fps_lbl = QLabel("FPS: 0")
        status_layout.addWidget(self.fps_lbl)
        
        top_layout.addWidget(self.camera_label, stretch=1)
        top_layout.addWidget(status_panel)
        
        # Bottom: High-contrast Subtitle Banner (What Speaker said)
        banner_frame = QFrame()
        banner_frame.setFixedHeight(120)
        banner_frame.setStyleSheet(f"background-color: #11111B; border: 2px solid {ACCENT_BLUE}; border-radius: {BORDER_RADIUS};")
        banner_layout = QVBoxLayout(banner_frame)
        self.subtitle_lbl = QLabel("Awaiting speech...")
        self.subtitle_lbl.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_lbl.setWordWrap(True)
        banner_layout.addWidget(self.subtitle_lbl)
        
        layout.addLayout(top_layout)
        layout.addWidget(banner_frame)
        return widget

    def _create_speaker_view(self) -> QWidget:
        """Create the layout for Hearing users."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Transcript View
        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setFont(QFont("Segoe UI", 16))
        self.transcript_area.setStyleSheet(f"background-color: {PANEL_COLOR}; padding: 10px;")
        
        # Bottom Controls
        controls = QHBoxLayout()
        self.reply_input = QLineEdit()
        self.reply_input.setPlaceholderText("Type reply or use microphone...")
        self.reply_input.setFont(QFont("Segoe UI", 14))
        self.reply_input.returnPressed.connect(self._send_reply)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_reply)
        
        self.mic_btn = QPushButton("🎤 Hold to Speak")
        # In a full implementation, we'd wire up local pyaudio capture here
        
        controls.addWidget(self.reply_input, stretch=1)
        controls.addWidget(self.send_btn)
        controls.addWidget(self.mic_btn)
        
        layout.addWidget(QLabel("Live Translation Feed (from Signer):"))
        layout.addWidget(self.transcript_area)
        layout.addLayout(controls)
        return widget

    def _start_workers(self):
        """Initialize and start background workers."""
        # 1. Network Worker
        self.network_worker = NetworkWorker(self.server_url, self.room_id, self.mode.lower())
        self.network_worker.connection_status.connect(self._on_connection_status)
        self.network_worker.message_received.connect(self._on_network_message)
        self.network_worker.start()
        
        # 2. Camera Worker (Only if Signer)
        if self.mode.lower() == "signer":
            self.camera_worker = CameraWorker()
            self.camera_worker.frame_ready.connect(self._update_camera_feed)
            self.camera_worker.sign_detected.connect(self._on_sign_detected)
            self.camera_worker.fps_updated.connect(self._update_fps)
            self.camera_worker.start()

    def _reconnect(self):
        """Restart network worker with new room ID."""
        new_room = self.room_input.text().strip()
        if not new_room:
            return
            
        self.room_id = new_room
        if self.network_worker:
            self.network_worker.stop()
            
        self.status_badge.setText("🟡 Reconnecting...")
        self.status_badge.setStyleSheet("color: yellow;")
        self.network_worker = NetworkWorker(self.server_url, self.room_id, self.mode.lower())
        self.network_worker.connection_status.connect(self._on_connection_status)
        self.network_worker.message_received.connect(self._on_network_message)
        self.network_worker.start()

    def _change_mode(self, new_mode: str):
        """Switch between Signer and Speaker modes."""
        self.mode = new_mode.lower()
        self.setWindowTitle(f"IsharaConnect (ইশারা কানেক্ট) - {new_mode} Mode")
        
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
            
        if self.mode == "signer":
            self.stacked_widget.setCurrentWidget(self.signer_view)
            self.camera_worker = CameraWorker()
            self.camera_worker.frame_ready.connect(self._update_camera_feed)
            self.camera_worker.sign_detected.connect(self._on_sign_detected)
            self.camera_worker.fps_updated.connect(self._update_fps)
            self.camera_worker.start()
        else:
            self.stacked_widget.setCurrentWidget(self.speaker_view)
            
        self._reconnect() # Reconnect network with new client_type

    @pyqtSlot(bool, str)
    def _on_connection_status(self, connected: bool, message: str):
        if connected:
            self.status_badge.setText(f"🟢 Connected: {self.room_id}")
            self.status_badge.setStyleSheet(f"color: {ACCENT_GREEN};")
        else:
            self.status_badge.setText("🔴 Disconnected")
            self.status_badge.setStyleSheet(f"color: {ACCENT_RED};")
            
        logger.info("Network Status: %s", message)

    @pyqtSlot(QImage)
    def _update_camera_feed(self, image: QImage):
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

    @pyqtSlot(float)
    def _update_fps(self, fps: float):
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

    @pyqtSlot(dict)
    def _on_sign_detected(self, data: dict):
        # Update local UI
        label = data.get("label_bn", "")
        conf = data.get("confidence", 0.0)
        
        self.local_sign_lbl.setText(label)
        self.conf_bar.setValue(int(conf * 100))
        
        # Send to backend
        if self.network_worker and data.get("is_stable"):
            self.network_worker.send_sign_event(data)

    @pyqtSlot(dict)
    def _on_network_message(self, data: dict):
        event_type = data.get("type")
        payload = data.get("data", {})
        
        if event_type == "SIGN_TRANSLATION" and self.mode == "speaker":
            text = payload.get("label_bn", "")
            self.transcript_area.append(f"<b>Signer:</b> {text}")
            
            # Optional: Play TTS audio if audio_payload_base64 exists
            # import base64, io, pydub ...
            
        elif event_type == "SPEECH_TEXT" and self.mode == "signer":
            transcript = payload.get("transcript", "")
            self.subtitle_lbl.setText(transcript)

    def _send_reply(self):
        """Send typed text from speaker to signer."""
        text = self.reply_input.text().strip()
        if text and self.network_worker:
            self.network_worker.send_speech_event(text)
            self.transcript_area.append(f"<span style='color:{ACCENT_BLUE}'><b>You:</b> {text}</span>")
            self.reply_input.clear()

    @pyqtSlot()
    def _open_propose_dialog(self):
        """Open the sign proposal dialog."""
        dialog = ProposeSignDialog(server_url=self.server_url)
        # Pause camera worker if signer, as the dialog uses the camera
        was_running = False
        if self.camera_worker:
            was_running = True
            self.camera_worker.stop()
            self.camera_worker = None
            
        dialog.exec()
        
        # Resume camera worker
        if was_running and self.mode == "signer":
            self.camera_worker = CameraWorker()
            self.camera_worker.frame_ready.connect(self._update_camera_feed)
            self.camera_worker.sign_detected.connect(self._on_sign_detected)
            self.camera_worker.fps_updated.connect(self._update_fps)
            self.camera_worker.start()

    def closeEvent(self, event):
        """Handle window close gracefully."""
        if self.camera_worker:
            self.camera_worker.stop()
        if self.network_worker:
            self.network_worker.stop()
        event.accept()
