"""Modern PyQt6 MainWindow for IsharaConnect Desktop Client."""

import logging
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QBrush, QPen
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
from desktop_app.ui.academy_dashboard import AcademyDashboard
from desktop_app.ui.scenario_simulator import ScenarioSimulator
from core_engine.audio.audio_player import player_instance

logger = logging.getLogger(__name__)

from desktop_app.ui.theme import ThemeStyles
from desktop_app.ui.components.badges import PulsingStatusBadge
from desktop_app.ui.components.sentence_ticker import SentenceTickerWidget
from desktop_app.ui.components.motion_trajectory_viewer import MotionTrajectoryViewer
from desktop_app.ui.components.gesture_avatar import GestureAvatarWidget

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
        self.setStyleSheet(ThemeStyles.get_global_stylesheet())
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Header ---
        header_layout = QHBoxLayout()
        title_label = QLabel("IsharaConnect")
        title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #06B6D4;")
        
        self.status_badge = PulsingStatusBadge("Offline", "#F43F5E")
        
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
        self.nav_mode.addItems(["💬 Communication Mode", "🎓 Learning Hub", "🏫 BdSL Academy", "🚑 Scenario Simulator"])
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
        self.academy_view = AcademyDashboard()
        self.scenario_view = ScenarioSimulator()
        
        self.stacked_widget.addWidget(self.signer_view)
        self.stacked_widget.addWidget(self.speaker_view)
        self.stacked_widget.addWidget(self.learning_view)
        self.stacked_widget.addWidget(self.academy_view)
        self.stacked_widget.addWidget(self.scenario_view)
        
        if self.mode.lower() == "signer":
            self.stacked_widget.setCurrentWidget(self.signer_view)
        else:
            self.stacked_widget.setCurrentWidget(self.speaker_view)
            
        main_layout.addWidget(self.stacked_widget)
        
        # Connect Learning Hub Signals
        self.learning_view.request_camera_start.connect(self._ensure_camera_running)
        self.learning_view.request_camera_stop.connect(self._stop_camera)
        self.academy_view.request_back.connect(lambda: self.nav_mode.setCurrentIndex(0))
        self.scenario_view.request_back.connect(lambda: self.nav_mode.setCurrentIndex(0))

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
        
        self.comm_view = QWidget()
        comm_layout = QVBoxLayout(self.comm_view)
        
        # Upper area: Camera and Chat
        upper_comm = QHBoxLayout()
        
        # Signer View (Camera with Motion Trajectory Viewer overlay)
        cam_container = QWidget()
        cam_container.setFixedSize(640, 480)
        
        self.camera_feed = QLabel(cam_container)
        self.camera_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_feed.setFixedSize(640, 480)
        self.camera_feed.setObjectName("GlassCard")
        
        self.trajectory_viewer = MotionTrajectoryViewer(cam_container)
        self.trajectory_viewer.setFixedSize(640, 480)
        
        upper_comm.addWidget(cam_container)
        
        # Speaker View & Visual Avatar
        side_layout = QVBoxLayout()
        self.gesture_avatar = GestureAvatarWidget()
        side_layout.addWidget(self.gesture_avatar)
        
        self.chat_history = QTextBrowser()
        self.chat_history.setObjectName("GlassCard")
        self.chat_history.setMaximumHeight(160)
        side_layout.addWidget(self.chat_history)
        
        reply_layout = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type speech to synthesize BdSL gestures...")
        self.text_input.returnPressed.connect(self._on_send_text)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send_text)
        
        reply_layout.addWidget(self.text_input)
        reply_layout.addWidget(self.send_btn)
        side_layout.addLayout(reply_layout)
        upper_comm.addLayout(side_layout)
        
        comm_layout.addLayout(upper_comm)
        
        # Lower area: Sentence Ticker HUD
        self.sentence_ticker = SentenceTickerWidget()
        comm_layout.addWidget(self.sentence_ticker)
        return self.comm_view

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
        is_academy = "Academy" in new_mode_str
        is_scenario = "Scenario" in new_mode_str
        
        if is_learning or is_academy or is_scenario:
            if is_learning:
                self.stacked_widget.setCurrentWidget(self.learning_view)
            elif is_academy:
                self.stacked_widget.setCurrentWidget(self.academy_view)
            elif is_scenario:
                self.stacked_widget.setCurrentWidget(self.scenario_view)
                
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
            self.camera_worker.trajectory_ready.connect(self._on_trajectory_ready)
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
    def _on_network_connected(self):
        self.status_badge.set_status("Online", "#10B981")
        self.connect_btn.setText("Disconnect")
        
    def _on_network_disconnected(self):
        self.status_badge.set_status("Offline", "#F43F5E")
        self.connect_btn.setText("Connect")

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
        elif hasattr(self, 'camera_feed'):
            scaled = pixmap.scaled(self.camera_feed.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.camera_feed.setPixmap(scaled)

    @pyqtSlot(float)
    def _update_fps(self, fps: float):
        self.fps_lbl.setText(f"FPS: {fps:.1f}")

    @pyqtSlot(dict)
    def _on_trajectory_ready(self, data: dict):
        if hasattr(self, 'trajectory_viewer'):
            self.trajectory_viewer.update_trajectory(
                left_wrist=data.get("left_wrist"),
                right_wrist=data.get("right_wrist"),
                left_index=data.get("left_index"),
                right_index=data.get("right_index")
            )

    @pyqtSlot(dict)
    def _on_sign_detected(self, data: dict):
        if self.stacked_widget.currentWidget() == self.learning_view:
            self.learning_view.process_prediction(data)
            return

        # Update Sentence Ticker
        if hasattr(self, 'sentence_ticker'):
            self.sentence_ticker.update_ticker(data)
        
        # Send to backend
        if self.network_worker and data.get("is_stable"):
            try:
                self.network_worker.send_sign_event(data)
            except Exception as e:
                logger.warning("MainWindow: Failed to forward sign event to NetworkWorker: %s", e)

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
            if hasattr(self, 'subtitle_lbl'):
                self.subtitle_lbl.setText(transcript)
            if hasattr(self, 'gesture_avatar'):
                self.gesture_avatar.synthesize_and_play(transcript)
            if hasattr(self, 'chat_history'):
                self.chat_history.append(f"<span style='color:#06B6D4'><b>Speaker:</b> {transcript}</span>")
            
    def _on_send_text(self):
        """Send typed text, play visual gesture avatar, and transmit to network."""
        text = self.text_input.text().strip()
        if text:
            if hasattr(self, 'gesture_avatar'):
                self.gesture_avatar.synthesize_and_play(text)
            if self.network_worker:
                self.network_worker.send_speech_event(text)
            self.chat_history.append(f"<span style='color:#10B981'><b>You:</b> {text}</span>")
            self.text_input.clear()
            
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
