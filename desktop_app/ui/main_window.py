"""Modern PyQt6 MainWindow for IsharaConnect Desktop Client."""

import logging
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QLineEdit, QComboBox, 
    QFrame, QStackedWidget, QTextEdit, QProgressBar,
    QTextBrowser, QCheckBox
)

from desktop_app.controllers.camera_worker import CameraWorker
from desktop_app.controllers.network_worker import NetworkWorker
from desktop_app.ui.propose_sign_dialog import ProposeSignDialog
from desktop_app.ui.learning_hub import LearningHubWidget
from core_engine.audio.audio_player import player_instance

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
        self.audio_cache = {}
        
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
        
        # Top-level navigation
        self.nav_mode = QComboBox()
        self.nav_mode.addItems(["💬 Communication Mode", "🎓 Learning Hub"])
        self.nav_mode.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.nav_mode.currentTextChanged.connect(self._change_app_mode)
        header_layout.addWidget(self.nav_mode)
        
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
        self.learning_view = LearningHubWidget()
        
        self.stacked_widget.addWidget(self.signer_view)
        self.stacked_widget.addWidget(self.speaker_view)
        self.stacked_widget.addWidget(self.learning_view)
        
        if self.mode.lower() == "signer":
            self.stacked_widget.setCurrentWidget(self.signer_view)
        else:
            self.stacked_widget.setCurrentWidget(self.speaker_view)
            
        main_layout.addWidget(self.stacked_widget)
        
        # Connect Learning Hub Signals
        self.learning_view.request_camera_start.connect(self._ensure_camera_running)
        self.learning_view.request_camera_stop.connect(self._stop_camera)

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
        
        status_layout.addWidget(QLabel("Camera Device:"))
        self.cam_selector = QComboBox()
        self.cam_selector.addItems(["Auto (0)", "Camera 1", "Camera 2", "Camera 3"])
        self.cam_selector.currentIndexChanged.connect(self._change_camera)
        status_layout.addWidget(self.cam_selector)
        
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
        self.transcript_area = QTextBrowser()
        self.transcript_area.setOpenLinks(False)
        self.transcript_area.anchorClicked.connect(self._on_play_voice_clicked)
        self.transcript_area.setFont(QFont("Segoe UI", 16))
        self.transcript_area.setStyleSheet(f"background-color: {PANEL_COLOR}; padding: 10px;")
        
        # Audio Options
        audio_layout = QHBoxLayout()
        self.auto_play_cb = QCheckBox("Auto-Play TTS")
        self.auto_play_cb.setChecked(True)
        self.auto_play_cb.setStyleSheet(f"color: {TEXT_COLOR}; font-weight: bold;")
        audio_layout.addWidget(self.auto_play_cb)
        audio_layout.addStretch()
        
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
        layout.addLayout(audio_layout)
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
            self.camera_worker.error_occurred.connect(self._on_camera_error)
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

    def _change_app_mode(self, new_mode_str: str):
        """Switch between Communication Mode and Learning Hub."""
        is_learning = "Learning" in new_mode_str
        
        if is_learning:
            self.stacked_widget.setCurrentWidget(self.learning_view)
            self.status_badge.hide()
            self.room_input.hide()
            self.join_btn.hide()
            self.mode_selector.hide()
            self.propose_btn.hide()
            if self.network_worker:
                self.network_worker.stop()
        else:
            if self.mode.lower() == "signer":
                self.stacked_widget.setCurrentWidget(self.signer_view)
            else:
                self.stacked_widget.setCurrentWidget(self.speaker_view)
            self.status_badge.show()
            self.room_input.show()
            self.join_btn.show()
            self.mode_selector.show()
            self.propose_btn.show()
            self._reconnect()
            self._ensure_camera_running()
            
    def _ensure_camera_running(self):
        """Starts camera if not already running (for Signer mode or Learning Hub)."""
        idx = self.cam_selector.currentIndex() if hasattr(self, 'cam_selector') else 0
        if not self.camera_worker:
            self.camera_worker = CameraWorker(camera_id=idx)
            self.camera_worker.frame_ready.connect(self._update_camera_feed)
            self.camera_worker.sign_detected.connect(self._on_sign_detected)
            self.camera_worker.fps_updated.connect(self._update_fps)
            self.camera_worker.error_occurred.connect(self._on_camera_error)
            self.camera_worker.start()

    def _stop_camera(self):
        """Stops camera."""
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None

    def _change_mode(self, new_mode: str):
        """Switch between Signer and Speaker modes."""
        self.mode = new_mode.lower()
        self.setWindowTitle(f"IsharaConnect (ইশারা কানেক্ট) - {new_mode} Mode")
        
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
            
        if self.mode == "signer":
            self.stacked_widget.setCurrentWidget(self.signer_view)
            self._ensure_camera_running()
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

    @pyqtSlot(str)
    def _on_camera_error(self, message: str):
        self.status_badge.setText(f"⚠️ Cam Error")
        self.status_badge.setStyleSheet("color: yellow;")
        self.camera_label.setText(message)
        logger.error("Camera Error: %s", message)

    def _change_camera(self, index: int):
        """Restart camera worker with new index."""
        if not self.camera_worker or self.mode != "signer":
            return
            
        self.camera_worker.stop()
        self.camera_label.setText("Initializing Camera...")
        self.camera_worker = CameraWorker(camera_id=index)
        self.camera_worker.frame_ready.connect(self._update_camera_feed)
        self.camera_worker.sign_detected.connect(self._on_sign_detected)
        self.camera_worker.fps_updated.connect(self._update_fps)
        self.camera_worker.error_occurred.connect(self._on_camera_error)
        self.camera_worker.start()

    @pyqtSlot(QImage)
    def _update_camera_feed(self, image: QImage):
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(image)
        # Route to the appropriate view
        if self.stacked_widget.currentWidget() == self.learning_view:
            self.learning_view.update_camera_feed(image)
        elif hasattr(self, 'camera_label'):
            scaled = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.camera_label.setPixmap(scaled)

    @pyqtSlot(float)
    def _update_fps(self, fps: float):
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

    @pyqtSlot(dict)
    def _on_sign_detected(self, data: dict):
        if self.stacked_widget.currentWidget() == self.learning_view:
            self.learning_view.process_prediction(data)
            return

        # Update local UI for Communication Mode
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
            player_instance.play_chime("notify")
            text = payload.get("label_bn", "")
            base64_audio = payload.get("audio_payload_base64", "")
            
            if base64_audio:
                msg_id = len(self.audio_cache)
                self.audio_cache[msg_id] = base64_audio
                self.transcript_area.append(f"<b>Signer:</b> {text} &nbsp;&nbsp; <a href='play:{msg_id}' style='color:{ACCENT_GREEN}; text-decoration:none;'>🔊 Play Voice</a>")
                
                if self.auto_play_cb.isChecked():
                    player_instance.play_base64(base64_audio)
            else:
                self.transcript_area.append(f"<b>Signer:</b> {text}")
            
        elif event_type == "SPEECH_TEXT" and self.mode == "signer":
            player_instance.play_chime("notify")
            transcript = payload.get("transcript", "")
            self.subtitle_lbl.setText(transcript)
            
    def _on_play_voice_clicked(self, url):
        """Handle inline play voice links."""
        scheme = url.scheme()
        if scheme == "play":
            try:
                msg_id = int(url.path())
                if msg_id in self.audio_cache:
                    player_instance.play_base64(self.audio_cache[msg_id])
            except ValueError:
                pass

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
            idx = self.cam_selector.currentIndex() if hasattr(self, 'cam_selector') else 0
            self.camera_worker = CameraWorker(camera_id=idx)
            self.camera_worker.frame_ready.connect(self._update_camera_feed)
            self.camera_worker.sign_detected.connect(self._on_sign_detected)
            self.camera_worker.fps_updated.connect(self._update_fps)
            self.camera_worker.error_occurred.connect(self._on_camera_error)
            self.camera_worker.start()

    def closeEvent(self, event):
        """Handle window close gracefully."""
        if self.camera_worker:
            self.camera_worker.stop()
        if self.network_worker:
            self.network_worker.stop()
        event.accept()
