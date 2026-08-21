"""Visual Gesture Avatar Player Widget for Speech-to-BdSL Playback.

Renders animated sequences of BdSL gesture flashcards with playback controls,
timeline scrubbing, and automated finger-spelling visualization.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtSvgWidgets import QSvgWidget

from core_engine.vision.gesture_synthesizer import BdSLGestureSynthesizer
from desktop_app.ui.theme import ThemeColors
import logging

logger = logging.getLogger(__name__)


class GestureAvatarWidget(QWidget):
    """Interactive visual BdSL gesture player translating speech/text into animated sign avatars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.synthesizer = BdSLGestureSynthesizer()
        self.gesture_sequence = []
        self.current_index = 0
        self.playback_speed = 1.0
        self.is_playing = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self.setObjectName("GlassCard")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header Title & Badges
        header_layout = QHBoxLayout()
        avatar_title = QLabel("🤖 BdSL Visual Avatar")
        avatar_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        avatar_title.setStyleSheet(f"color: {ThemeColors.CYAN_ACCENT};")

        self.badge_lbl = QLabel("Ready")
        self.badge_lbl.setStyleSheet(f"""
            background-color: {ThemeColors.SURFACE_DARK};
            color: {ThemeColors.EMERALD_SUCCESS};
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 11px;
            font-weight: bold;
        """)

        header_layout.addWidget(avatar_title)
        header_layout.addStretch()
        header_layout.addWidget(self.badge_lbl)
        layout.addLayout(header_layout)

        # Central Visual Card Display
        self.card_container = QWidget()
        self.card_container.setFixedHeight(260)
        self.card_container.setStyleSheet(f"background-color: {ThemeColors.BG_DARK}; border-radius: 10px;")
        card_layout = QVBoxLayout(self.card_container)
        card_layout.setContentsMargins(5, 5, 5, 5)

        # SVG Renderer
        self.svg_viewer = QSvgWidget()
        self.svg_viewer.setFixedHeight(250)
        card_layout.addWidget(self.svg_viewer)

        # Text Fallback Label (if SVG missing)
        self.text_card_lbl = QLabel("Enter speech/text to generate BdSL gestures")
        self.text_card_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_card_lbl.setFont(QFont("SolaimanLipi", 16))
        self.text_card_lbl.setStyleSheet(f"color: {ThemeColors.TEXT_SECONDARY};")
        self.text_card_lbl.hide()
        card_layout.addWidget(self.text_card_lbl)

        layout.addWidget(self.card_container)

        # Subtitle Information Area
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        self.sign_name_lbl = QLabel("---")
        self.sign_name_lbl.setFont(QFont("SolaimanLipi", 18, QFont.Weight.Bold))
        self.sign_name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_name_lbl.setStyleSheet(f"color: {ThemeColors.TEXT_PRIMARY};")

        self.sign_en_lbl = QLabel("Awaiting input...")
        self.sign_en_lbl.setFont(QFont("Inter", 12))
        self.sign_en_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_en_lbl.setStyleSheet(f"color: {ThemeColors.TEXT_SECONDARY};")

        info_layout.addWidget(self.sign_name_lbl)
        info_layout.addWidget(self.sign_en_lbl)
        layout.addLayout(info_layout)

        # Timeline Progress Bar
        self.timeline_bar = QProgressBar()
        self.timeline_bar.setRange(0, 100)
        self.timeline_bar.setValue(0)
        self.timeline_bar.setTextVisible(False)
        self.timeline_bar.setFixedHeight(6)
        self.timeline_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {ThemeColors.SURFACE_DARK};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {ThemeColors.CYAN_ACCENT};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.timeline_bar)

        # Playback Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedHeight(32)
        self.play_btn.clicked.connect(self._toggle_playback)

        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setFixedHeight(32)
        self.reset_btn.setStyleSheet(f"background-color: {ThemeColors.SURFACE_DARK}; color: {ThemeColors.TEXT_PRIMARY};")
        self.reset_btn.clicked.connect(self.reset)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x Speed", "1.0x Speed", "1.5x Speed", "2.0x Speed"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._change_speed)
        self.speed_combo.setFixedHeight(32)

        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.speed_combo)
        layout.addLayout(controls_layout)

    @pyqtSlot(str)
    def synthesize_and_play(self, text: str):
        """Translates input text into gesture frames and starts playback."""
        if not text or not text.strip():
            return

        self.gesture_sequence = self.synthesizer.synthesize_text_to_gestures(
            text, speed=self.playback_speed
        )

        if not self.gesture_sequence:
            self.sign_name_lbl.setText("No signs resolved")
            return

        self.current_index = 0
        self.is_playing = True
        self.play_btn.setText("⏸ Pause")
        self.badge_lbl.setText(f"Playing ({len(self.gesture_sequence)} Signs)")
        self.badge_lbl.setStyleSheet(f"background-color: {ThemeColors.SURFACE_DARK}; color: {ThemeColors.CYAN_ACCENT}; border-radius: 6px; padding: 3px 8px; font-size: 11px; font-weight: bold;")
        
        self._display_current_gesture()

    def _display_current_gesture(self):
        """Renders the current gesture frame."""
        if not self.gesture_sequence or self.current_index >= len(self.gesture_sequence):
            self._playback_finished()
            return

        frame = self.gesture_sequence[self.current_index]
        self.sign_name_lbl.setText(frame.get("label_bn", ""))
        self.sign_en_lbl.setText(f"{frame.get('label_en', '')} | {frame.get('motion_type', '')}")

        # Update progress bar
        progress = int(((self.current_index + 1) / len(self.gesture_sequence)) * 100)
        self.timeline_bar.setValue(progress)

        # Load SVG Card
        card_path = frame.get("card_path")
        if card_path and card_path.endswith(".svg"):
            try:
                self.svg_viewer.load(card_path)
                self.svg_viewer.show()
                self.text_card_lbl.hide()
            except Exception as e:
                logger.warning("Failed to render SVG: %s", e)
                self.svg_viewer.hide()
                self.text_card_lbl.setText(frame.get("label_bn", ""))
                self.text_card_lbl.show()
        else:
            self.svg_viewer.hide()
            self.text_card_lbl.setText(f"Sign: {frame.get('label_bn', '')}\n({frame.get('label_en', '')})")
            self.text_card_lbl.show()

        duration = frame.get("duration_ms", 800)
        self.timer.start(duration)

    def _advance_frame(self):
        """Advances to the next gesture frame."""
        self.timer.stop()
        if not self.is_playing:
            return

        self.current_index += 1
        if self.current_index < len(self.gesture_sequence):
            self._display_current_gesture()
        else:
            self._playback_finished()

    def _playback_finished(self):
        """Called when gesture animation completes."""
        self.is_playing = False
        self.timer.stop()
        self.play_btn.setText("▶ Replay")
        self.badge_lbl.setText("Finished")
        self.badge_lbl.setStyleSheet(f"background-color: {ThemeColors.SURFACE_DARK}; color: {ThemeColors.EMERALD_SUCCESS}; border-radius: 6px; padding: 3px 8px; font-size: 11px; font-weight: bold;")
        self.timeline_bar.setValue(100)

    def _toggle_playback(self):
        """Toggles between Play and Pause."""
        if not self.gesture_sequence:
            return

        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.play_btn.setText("▶ Play")
            self.badge_lbl.setText("Paused")
        else:
            if self.current_index >= len(self.gesture_sequence):
                self.current_index = 0
            self.is_playing = True
            self.play_btn.setText("⏸ Pause")
            self.badge_lbl.setText("Playing")
            self._display_current_gesture()

    def reset(self):
        """Resets the player to initial frame."""
        self.timer.stop()
        self.is_playing = False
        self.current_index = 0
        self.play_btn.setText("▶ Play")
        self.badge_lbl.setText("Ready")
        if self.gesture_sequence:
            self._display_current_gesture()
            self.timer.stop()  # Keep static until user clicks play
        self.timeline_bar.setValue(0)

    def _change_speed(self, index: int):
        """Updates playback speed multiplier."""
        speeds = [0.5, 1.0, 1.5, 2.0]
        if 0 <= index < len(speeds):
            self.playback_speed = speeds[index]
