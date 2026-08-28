"""Interactive Academy Avatar Playback Controller Widget.

Provides cybernetic-styled playback controls for ToonAvatarRenderer:
  - ⏯️ Play / Pause toggle
  - ⏱️ Speed selector: 1.0x (Normal), 0.5x (Half speed), 0.25x (Ultra slow-motion)
  - 🔄 Continuous Loop toggle
  - 🔍 Perspective Viewport toggle (Full Body vs Hand Close-Up Zoom)
  - ⏩ Frame scrubber / Step-by-step slider
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from desktop_app.ui.components.toon_avatar_renderer import ToonAvatarRenderer

logger = logging.getLogger(__name__)

# Cyberpunk Modern Palette
BG_CARD       = "rgba(15, 23, 42, 0.95)"
BORDER_GLOW   = "rgba(6, 182, 212, 0.35)"
CYAN_ACCENT   = "#06B6D4"
EMERALD_GREEN = "#10B981"
AMBER_YELLOW  = "#F59E0B"
TEXT_MAIN     = "#F8FAFC"
TEXT_MUTED    = "#94A3B8"
SURFACE_BTN   = "#1E293B"


class AvatarPlaybackBar(QFrame):
    """Modern cybernetic styled playback bar with speed buttons, scrub bar, and zoom toggle."""

    play_toggled = pyqtSignal(bool)       # is_playing
    speed_changed = pyqtSignal(float)     # 1.0, 0.5, 0.25
    loop_toggled = pyqtSignal(bool)       # is_loop
    frame_seeked = pyqtSignal(int)        # target_frame
    zoom_toggled = pyqtSignal(str)        # "full_body" or "hand_zoom"

    def __init__(self, avatar_renderer: Optional[ToonAvatarRenderer] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.avatar = avatar_renderer
        self._is_playing = True
        self._is_loop = True
        self._current_speed = 1.0
        self._is_hand_zoom = False
        self._total_frames = 60
        self._current_frame = 0

        self._init_ui()

        if self.avatar is not None:
            self.attach_avatar(self.avatar)

    def _init_ui(self) -> None:
        self.setObjectName("AvatarPlaybackBar")
        self.setStyleSheet(f"""
            QFrame#AvatarPlaybackBar {{
                background: {BG_CARD};
                border: 1px solid {BORDER_GLOW};
                border-radius: 10px;
                padding: 6px 8px;
            }}
            QPushButton {{
                background: {SURFACE_BTN};
                color: {TEXT_MAIN};
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 600;
                min-height: 34px;
            }}
            QPushButton:hover {{
                background: #283548;
                border-color: {CYAN_ACCENT};
                color: {CYAN_ACCENT};
            }}
            QPushButton:checked {{
                background: {CYAN_ACCENT};
                color: #0F172A;
                border-color: {CYAN_ACCENT};
                font-weight: bold;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: rgba(255, 255, 255, 0.15);
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {CYAN_ACCENT};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: #FFFFFF;
                border: 2px solid {CYAN_ACCENT};
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # ── Row 1: Scrubber Timeline Slider + High-Contrast Time Code Badge ──
        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(8)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 59)
        self.slider.setValue(0)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        scrub_row.addWidget(self.slider, stretch=1)

        self.lbl_frame = QLabel("1/60")
        self.lbl_frame.setFont(QFont("JetBrains Mono", 11, QFont.Weight.Bold))
        self.lbl_frame.setStyleSheet(f"""
            color: #38BDF8;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 5px;
            padding: 3px 8px;
            min-width: 52px;
        """)
        self.lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scrub_row.addWidget(self.lbl_frame)

        layout.addLayout(scrub_row)

        # ── Row 2: Transport Controls & Speed Selectors ──────────────────────
        ctrl_row1 = QHBoxLayout()
        ctrl_row1.setSpacing(6)

        # 1. Play / Pause
        self.btn_play = QPushButton("⏸")
        self.btn_play.setMinimumWidth(38)
        self.btn_play.setFixedHeight(36)
        self.btn_play.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.btn_play.setToolTip("চালু / বিরতি (Play/Pause)")
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl_row1.addWidget(self.btn_play)

        # 2. Step Backward / Step Forward
        self.btn_prev = QPushButton("⏮")
        self.btn_prev.setMinimumWidth(34)
        self.btn_prev.setFixedHeight(36)
        self.btn_prev.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_prev.setToolTip("পূর্ববর্তী ফ্রেম")
        self.btn_prev.clicked.connect(self._step_back)
        ctrl_row1.addWidget(self.btn_prev)

        self.btn_next = QPushButton("⏭")
        self.btn_next.setMinimumWidth(34)
        self.btn_next.setFixedHeight(36)
        self.btn_next.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btn_next.setToolTip("পরবর্তী ফ্রেম")
        self.btn_next.clicked.connect(self._step_fwd)
        ctrl_row1.addWidget(self.btn_next)

        # 3. Speed Buttons (1.0x, 0.5x, 0.25x) - Min width 54px, Height 36px, 12px Font
        self.btn_spd_1x = QPushButton("1.0x")
        self.btn_spd_1x.setMinimumWidth(54)
        self.btn_spd_1x.setFixedHeight(36)
        self.btn_spd_1x.setCheckable(True)
        self.btn_spd_1x.setChecked(True)
        self.btn_spd_1x.clicked.connect(lambda: self._set_speed(1.0))
        ctrl_row1.addWidget(self.btn_spd_1x)

        self.btn_spd_05x = QPushButton("0.5x")
        self.btn_spd_05x.setMinimumWidth(54)
        self.btn_spd_05x.setFixedHeight(36)
        self.btn_spd_05x.setCheckable(True)
        self.btn_spd_05x.clicked.connect(lambda: self._set_speed(0.5))
        ctrl_row1.addWidget(self.btn_spd_05x)

        self.btn_spd_025x = QPushButton("0.25x")
        self.btn_spd_025x.setMinimumWidth(54)
        self.btn_spd_025x.setFixedHeight(36)
        self.btn_spd_025x.setCheckable(True)
        self.btn_spd_025x.clicked.connect(lambda: self._set_speed(0.25))
        ctrl_row1.addWidget(self.btn_spd_025x)

        layout.addLayout(ctrl_row1)

        # ── Row 3: Action Controls (Loop & Perspective Hand Zoom) ─────────────
        ctrl_row2 = QHBoxLayout()
        ctrl_row2.setSpacing(6)

        # 4. Continuous Loop Button
        self.btn_loop = QPushButton("🔄 লুপ")
        self.btn_loop.setMinimumWidth(64)
        self.btn_loop.setFixedHeight(36)
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(True)
        self.btn_loop.setToolTip("ক্রমাগত লুপ (Continuous Loop)")
        self.btn_loop.clicked.connect(self._toggle_loop)
        ctrl_row2.addWidget(self.btn_loop, stretch=1)

        # 5. Perspective Toggle Button (Hand Zoom vs Full Body)
        self.btn_zoom = QPushButton("🔍 হাত জুম")
        self.btn_zoom.setMinimumWidth(90)
        self.btn_zoom.setFixedHeight(36)
        self.btn_zoom.setCheckable(True)
        self.btn_zoom.setChecked(False)
        self.btn_zoom.setToolTip("ফুল বডি / হাতের ক্লোজ-আপ জুম (Hand Zoom)")
        self.btn_zoom.clicked.connect(self._toggle_zoom)
        ctrl_row2.addWidget(self.btn_zoom, stretch=1)

        layout.addLayout(ctrl_row2)

    # ── Avatar Binding ───────────────────────────────────────────────────────

    def attach_avatar(self, avatar: ToonAvatarRenderer) -> None:
        """Binds this playback bar to a ToonAvatarRenderer instance."""
        self.avatar = avatar
        self.avatar.frame_changed.connect(self.on_avatar_frame_changed)
        self.avatar.playback_state_changed.connect(self.on_avatar_playback_changed)
        self.avatar.zoom_mode_changed.connect(self.on_avatar_zoom_changed)
        self.slider.setRange(0, max(1, self.avatar.total_frames - 1))

    def on_avatar_frame_changed(self, cur: int, total: int) -> None:
        """Updates frame slider and label without firing feedback events."""
        self._current_frame = cur
        self._total_frames = max(1, total)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self._total_frames - 1)
        self.slider.setValue(cur)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"{cur + 1}/{self._total_frames}")

    def on_avatar_playback_changed(self, is_playing: bool) -> None:
        self._is_playing = is_playing
        self.btn_play.setText("⏸" if is_playing else "▶")

    def on_avatar_zoom_changed(self, mode: str) -> None:
        self._is_hand_zoom = (mode == "hand_zoom")
        self.btn_zoom.setChecked(self._is_hand_zoom)
        self.btn_zoom.setText("👤 ফুল বডি" if self._is_hand_zoom else "🔍 হাত জুম")

    # ── Event Handlers ───────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self.avatar is not None:
            self._is_playing = self.avatar.toggle_play()
        else:
            self._is_playing = not self._is_playing
            self.play_toggled.emit(self._is_playing)
        self.btn_play.setText("⏸" if self._is_playing else "▶")

    def _step_fwd(self) -> None:
        if self.avatar is not None:
            self.avatar.step_forward()
        else:
            nxt = (self._current_frame + 1) % self._total_frames
            self.frame_seeked.emit(nxt)

    def _step_back(self) -> None:
        if self.avatar is not None:
            self.avatar.step_backward()
        else:
            prv = (self._current_frame - 1 + self._total_frames) % self._total_frames
            self.frame_seeked.emit(prv)

    def _set_speed(self, spd: float) -> None:
        self._current_speed = spd
        self.btn_spd_1x.setChecked(spd == 1.0)
        self.btn_spd_05x.setChecked(spd == 0.5)
        self.btn_spd_025x.setChecked(spd == 0.25)

        if self.avatar is not None:
            self.avatar.set_speed(spd)
        self.speed_changed.emit(spd)

    def _toggle_loop(self) -> None:
        self._is_loop = self.btn_loop.isChecked()
        if self.avatar is not None:
            self.avatar.set_loop(self._is_loop)
        self.loop_toggled.emit(self._is_loop)

    def _toggle_zoom(self) -> None:
        if self.avatar is not None:
            mode = self.avatar.toggle_view_mode()
            self._is_hand_zoom = (mode == "hand_zoom")
        else:
            self._is_hand_zoom = not self._is_hand_zoom
            mode = "hand_zoom" if self._is_hand_zoom else "full_body"
            self.zoom_toggled.emit(mode)
        self.btn_zoom.setText("👤 ফুল বডি" if self._is_hand_zoom else "🔍 হাত জুম")

    def _on_slider_moved(self, val: int) -> None:
        if self.avatar is not None:
            self.avatar.seek(val)
        self.frame_seeked.emit(val)
