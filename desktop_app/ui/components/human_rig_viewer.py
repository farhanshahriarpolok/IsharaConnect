"""Ultra-Lightweight 2D Kinematic Human Skeleton Animator & Gesture Motion Loop Viewer.

Renders upper-body human anatomy, glowing cyber limbs, and 21-landmark hand phalanges
with double-buffered QPainter, Catmull-Rom smoothed motion interpolation, and particle trails.
"""

import collections
import logging
from typing import List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QWidget

from core_engine.vision.kinematic_interpolator import (
    HAND_CONNECTIONS,
    KinematicJointFrame,
    KinematicMotionInterpolator,
)

logger = logging.getLogger(__name__)

# Cyberpunk Aesthetic Color Palette
COLOR_BG_START = QColor("#0B0F19")
COLOR_BG_END = QColor("#111827")
COLOR_BORDER = QColor(6, 182, 212, 120)
COLOR_GRID = QColor(56, 189, 248, 18)

COLOR_SILHOUETTE = QColor("#1E293B")
COLOR_SILHOUETTE_STROKE = QColor("#334155")
COLOR_VISOR = QColor(6, 182, 212, 220)

COLOR_ARM = QColor("#0284C7")
COLOR_FOREARM = QColor("#38BDF8")
COLOR_JOINT = QColor("#06B6D4")

COLOR_PALM = QColor("#0EA5E9")
COLOR_FINGER_BONE = QColor("#34D399")
COLOR_FINGER_JOINT = QColor("#10B981")

COLOR_TRAIL_PRIMARY = QColor(56, 189, 248)
COLOR_TRAIL_SECONDARY = QColor(16, 185, 129)


class HumanRigViewer(QWidget):
    """Hardware-accelerated, lightweight 2D Kinematic Human Skeleton Animator."""

    def __init__(
        self,
        sign_slug: str = "dhonnobad",
        label_bn: str = "ধন্যবাদ",
        label_en: str = "Thank you",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.sign_slug = sign_slug
        self.label_bn = label_bn
        self.label_en = label_en

        self.interpolator = KinematicMotionInterpolator()
        self.frames: List[KinematicJointFrame] = []
        self.current_frame_idx = 0
        self.is_playing = True
        self.fps = 30
        self.trail_history = collections.deque(maxlen=14)

        # 30 FPS non-blocking animation timer (33ms)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._advance_frame)

        self.setMinimumSize(280, 240)
        self.load_sign_motion(sign_slug, label_bn, label_en)
        self.timer.start()

    def load_sign_motion(self, sign_slug: str, label_bn: str = "", label_en: str = ""):
        """Loads and compiles 60-frame motion loop for the requested sign."""
        self.sign_slug = sign_slug or "dhonnobad"
        if label_bn:
            self.label_bn = label_bn
        if label_en:
            self.label_en = label_en

        self.frames = self.interpolator.resolve_motion_sequence(
            self.sign_slug,
            self.label_bn,
            self.label_en
        )
        self.current_frame_idx = 0
        self.trail_history.clear()
        self.update()

    def play(self):
        """Starts animation playback."""
        self.is_playing = True
        if not self.timer.isActive():
            self.timer.start(33)
        self.update()

    def pause(self):
        """Pauses animation playback."""
        self.is_playing = False
        self.timer.stop()
        self.update()

    def toggle_playback(self):
        """Toggles between Play and Pause states."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def reset(self):
        """Resets playback to frame 0."""
        self.current_frame_idx = 0
        self.trail_history.clear()
        self.update()

    def mousePressEvent(self, event):
        """Interactive click to toggle Play/Pause."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_playback()

    def _advance_frame(self):
        """Advances to next frame in 60-frame loop."""
        if not self.frames:
            return

        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
        cur_frame = self.frames[self.current_frame_idx]

        # Record right wrist position in trailing particle queue
        if cur_frame.is_right_active:
            self.trail_history.append((cur_frame.right_wrist[0], cur_frame.right_wrist[1]))
        elif cur_frame.is_left_active:
            self.trail_history.append((cur_frame.left_wrist[0], cur_frame.left_wrist[1]))

        self.update()

    def paintEvent(self, event):
        """Double-buffered vector rendering of cyberpunk human skeleton rig."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = float(self.width())
        h = float(self.height())

        # 1. Background Card & Cyber Grid
        self._paint_background(painter, w, h)

        if not self.frames:
            painter.setPen(QColor("#94A3B8"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "Loading Motion...")
            painter.end()
            return

        frame = self.frames[self.current_frame_idx]

        # Coordinate mapper: Normalized [0, 1] -> Canvas Pixels
        def to_px(pt: Tuple[float, float]) -> QPointF:
            return QPointF(pt[0] * w, pt[1] * h)

        # 2. Motion Trajectory Ghost Trail
        self._paint_motion_trail(painter, w, h)

        # 3. Torso Silhouette & Head
        self._paint_torso_and_head(painter, frame, to_px)

        # 4. Upper Limbs & Skeleton Bones
        self._paint_limbs(painter, frame, to_px)

        # 5. 21-Landmark Hand Phalanges
        self._paint_hand_landmarks(painter, frame, to_px)

        # 6. HUD Badges and Playback Indicator
        self._paint_hud_overlays(painter, w, h)

        painter.end()

    def _paint_background(self, painter: QPainter, w: float, h: float):
        """Paints rounded cyberpunk gradient background with subtle grid lines."""
        rect = QRectF(2, 2, w - 4, h - 4)
        bg_path = QPainterPath()
        bg_path.addRoundedRect(rect, 12, 12)

        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, COLOR_BG_START)
        grad.setColorAt(1.0, COLOR_BG_END)
        painter.fillPath(bg_path, QBrush(grad))
        painter.setPen(QPen(COLOR_BORDER, 1.2))
        painter.drawPath(bg_path)

        # Subtle Glowing Grid lines
        painter.setPen(QPen(COLOR_GRID, 1.0, Qt.PenStyle.DotLine))
        grid_step = 35.0
        x = grid_step
        while x < w:
            painter.drawLine(int(x), 4, int(x), int(h - 4))
            x += grid_step
        y = grid_step
        while y < h:
            painter.drawLine(4, int(y), int(w - 4), int(y))
            y += grid_step

    def _paint_torso_and_head(self, painter: QPainter, frame: KinematicJointFrame, to_px):
        """Paints stylized cyber torso silhouette and head visor."""
        hx, hy = to_px(frame.head).x(), to_px(frame.head).y()
        nx, ny = to_px(frame.neck).x(), to_px(frame.neck).y()
        cx, cy = to_px(frame.chest).x(), to_px(frame.chest).y()
        lsx, lsy = to_px(frame.left_shoulder).x(), to_px(frame.left_shoulder).y()
        rsx, rsy = to_px(frame.right_shoulder).x(), to_px(frame.right_shoulder).y()

        # Torso Silhouette polygon
        torso_path = QPainterPath()
        torso_path.moveTo(lsx, lsy)
        torso_path.lineTo(rsx, rsy)
        torso_path.lineTo(rsx - 15, self.height() * 0.88)
        torso_path.lineTo(lsx + 15, self.height() * 0.88)
        torso_path.closeSubpath()

        painter.fillPath(torso_path, QBrush(COLOR_SILHOUETTE))
        painter.setPen(QPen(COLOR_SILHOUETTE_STROKE, 1.5))
        painter.drawPath(torso_path)

        # Cyber Collarbone / Chest Line
        painter.setPen(QPen(QColor(56, 189, 248, 70), 1.5))
        painter.drawLine(int(lsx), int(lsy), int(nx), int(ny))
        painter.drawLine(int(nx), int(ny), int(rsx), int(rsy))
        painter.drawLine(int(nx), int(ny), int(cx), int(cy))

        # Head Oval
        painter.setBrush(QBrush(COLOR_SILHOUETTE))
        painter.setPen(QPen(COLOR_SILHOUETTE_STROKE, 1.5))
        head_rect = QRectF(hx - 18, hy - 22, 36, 44)
        painter.drawEllipse(head_rect)

        # Glowing Cyber Visor
        painter.setBrush(QBrush(COLOR_VISOR))
        painter.setPen(Qt.PenStyle.NoPen)
        visor_rect = QRectF(hx - 12, hy - 6, 24, 7)
        painter.drawRoundedRect(visor_rect, 3, 3)

    def _paint_limbs(self, painter: QPainter, frame: KinematicJointFrame, to_px):
        """Paints glowing cyber upper arm and forearm bone links."""
        ls = to_px(frame.left_shoulder)
        le = to_px(frame.left_elbow)
        lw = to_px(frame.left_wrist)

        rs = to_px(frame.right_shoulder)
        re = to_px(frame.right_elbow)
        rw = to_px(frame.right_wrist)

        # Left Arm Bones
        painter.setPen(QPen(COLOR_ARM, 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(ls, le)
        painter.setPen(QPen(COLOR_FOREARM, 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(le, lw)

        # Right Arm Bones
        painter.setPen(QPen(COLOR_ARM, 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(rs, re)
        painter.setPen(QPen(COLOR_FOREARM, 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(re, rw)

        # Joint Nodes (Shoulders & Elbows)
        for pt in [ls, rs, le, re]:
            painter.setBrush(QBrush(COLOR_JOINT))
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1.2))
            painter.drawEllipse(pt, 4.0, 4.0)

    def _paint_hand_landmarks(self, painter: QPainter, frame: KinematicJointFrame, to_px):
        """Paints 21 MediaPipe phalanges and glowing joint nodes for active hands."""
        for hand, is_active in [(frame.right_hand, frame.is_right_active), (frame.left_hand, frame.is_left_active)]:
            if not hand or len(hand) < 21:
                continue

            pts = [to_px(pt) for pt in hand]

            # 1. Draw Phalange Bones
            painter.setPen(QPen(COLOR_FINGER_BONE, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for i1, i2 in HAND_CONNECTIONS:
                if i1 < len(pts) and i2 < len(pts):
                    painter.drawLine(pts[i1], pts[i2])

            # 2. Draw Joint Nodes
            for idx, pt in enumerate(pts):
                if idx == 0:
                    # Wrist Root
                    painter.setBrush(QBrush(COLOR_PALM))
                    painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                    painter.drawEllipse(pt, 4.5, 4.5)
                elif idx in [4, 8, 12, 16, 20]:
                    # Fingertips (glowing pulse)
                    painter.setBrush(QBrush(COLOR_FINGER_JOINT))
                    painter.setPen(QPen(QColor(255, 255, 255, 240), 1.2))
                    painter.drawEllipse(pt, 3.5, 3.5)
                else:
                    # Finger MCP / PIP / DIP
                    painter.setBrush(QBrush(COLOR_FINGER_JOINT))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(pt, 2.2, 2.2)

    def _paint_motion_trail(self, painter: QPainter, w: float, h: float):
        """Paints fading particle ghost trail for dynamic hand motion paths."""
        if len(self.trail_history) < 2:
            return

        trail_len = len(self.trail_history)
        for i in range(trail_len - 1):
            p0 = self.trail_history[i]
            p1 = self.trail_history[i + 1]
            alpha = int((i + 1) / trail_len * 180)
            
            trail_color = QColor(COLOR_TRAIL_PRIMARY)
            trail_color.setAlpha(alpha)

            pen = QPen(trail_color, 2.5 * ((i + 1) / trail_len))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(p0[0] * w, p0[1] * h), QPointF(p1[0] * w, p1[1] * h))

    def _paint_hud_overlays(self, painter: QPainter, w: float, h: float):
        """Paints futuristic HUD badges and playback status."""
        # Top-Left Live Motion Badge
        painter.setBrush(QBrush(QColor(15, 23, 42, 200)))
        painter.setPen(QPen(QColor(6, 182, 212, 180), 1.0))
        painter.drawRoundedRect(QRectF(8, 8, 120, 20), 4, 4)

        painter.setPen(QColor("#38BDF8"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(QRectF(10, 8, 116, 20), Qt.AlignmentFlag.AlignCenter, "🎬 LIVE ACTION (30 FPS)")

        # Top-Right Play/Pause Icon
        icon_str = "⏸" if self.is_playing else "▶"
        painter.setPen(QColor("#10B981" if self.is_playing else "#F59E0B"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(w - 28, 8, 20, 20), Qt.AlignmentFlag.AlignCenter, icon_str)

        # Bottom Title & Subtitle Badge
        painter.setBrush(QBrush(QColor(15, 23, 42, 210)))
        painter.setPen(QPen(QColor(56, 189, 248, 100), 1.0))
        painter.drawRoundedRect(QRectF(8, h - 30, w - 16, 22), 6, 6)

        painter.setPen(QColor("#F8FAFC"))
        painter.setFont(QFont("SolaimanLipi", 11, QFont.Weight.Bold))
        title_text = f"{self.label_bn}  •  {self.label_en}"
        painter.drawText(QRectF(12, h - 30, w - 24, 22), Qt.AlignmentFlag.AlignCenter, title_text)
